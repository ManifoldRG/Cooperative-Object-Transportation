"""Portable subprocess driver for the 10xT/60s campaign. All paths relative
to this file; repo from env COT_REPO. Resume-aware (CSV rows + JSONs skipped).
Per-task hard caps kill hung tasks and record ABSTAIN:DriverTimeout."""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "baseline_30s", "tenxT_60s_verified_10pa.csv")
RES_DIR = os.path.join(HERE, "tenxT_results")
WORKER = os.path.join(HERE, "task_worker.py")
K = int(os.environ.get("WORKERS", "5"))
METHODS = ["decentralized_gd", "centralized_gd", "centralized_nlp_th"]
# repo root inferred from this file's location (scripts/tenxT/ -> repo);
# COT_REPO env var overrides.
REPO = os.environ.get("COT_REPO") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

sc = pd.read_csv(os.path.join(REPO, "CSF Runs", "results_20260823",
                              "baseline_large_20_scenarios_th.csv"))
sc = sc.groupby("thrust_angle_deg", group_keys=False).head(10)
wanted = [(int(r.global_scenario_id), m) for _, r in sc.iterrows() for m in METHODS]

done = set()
if os.path.exists(CSV):
    prev = pd.read_csv(CSV)
    done |= set(zip(prev.scenario_id.astype(int), prev.method))
os.makedirs(RES_DIR, exist_ok=True)
for f in os.listdir(RES_DIR):
    if f.endswith(".json"):
        sid, m = f[:-5].split("_", 1)
        done.add((int(sid), m))

todo = [t for t in wanted if t not in done]
print(f"{len(done)} done, {len(todo)} to run (workers={K})", flush=True)

# hard per-task wall caps: a single uncapped IPOPT call inside GD can run
# tens of minutes on pathological instances. At a 60 s budget that IS a
# failure — kill and record it.
# fuel objective makes cold inner solves ~10x slower than energy did:
# caps sized so a single legitimate cold solve at 10xT cannot be killed
CAP_S = {"decentralized_gd": 20 * 60, "centralized_gd": 10 * 60,
         "centralized_nlp_th": 10 * 60}

running = {}  # popen -> (sid, method, t0)
t_start = time.perf_counter()
finished = 0
while todo or running:
    while todo and len(running) < K:
        sid, m = todo.pop(0)
        p = subprocess.Popen([sys.executable, WORKER, str(sid), m, RES_DIR],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running[p] = (sid, m, time.perf_counter())
    time.sleep(5)
    for p in list(running):
        sid, m, t0 = running[p]
        age = time.perf_counter() - t0
        if p.poll() is None and age > CAP_S[m]:
            p.kill()
            p.wait()
            deg = float(sc[sc.global_scenario_id == sid].thrust_angle_deg.iloc[0])
            nst = int(sc[sc.global_scenario_id == sid].N.iloc[0]) * 10
            with open(os.path.join(RES_DIR, f"{sid}_{m}.json"), "w") as f:
                json.dump(dict(scenario_id=sid, thrust_angle_deg=deg, method=m,
                               n_steps=nst, time_limit_s=60.0,
                               wall_s=round(age, 1), cost_reported=None,
                               cost_rollout=None, V_rollout=None,
                               ipopt_status="ABSTAIN:DriverTimeout",
                               converged=""), f)
        if p.poll() is not None:
            running.pop(p)
            finished += 1
            have = os.path.exists(os.path.join(RES_DIR, f"{sid}_{m}.json"))
            print(f"[{finished}/{len(todo)+finished+len(running)}] {sid} {m} "
                  f"exit={p.returncode} json={'ok' if have else 'MISSING'} "
                  f"wall={age/60:.1f}min ({(time.perf_counter()-t_start)/60:.1f} min)",
                  flush=True)

# merge JSONs into the CSV
rows = []
for f in sorted(os.listdir(RES_DIR)):
    if f.endswith(".json"):
        rows.append(json.load(open(os.path.join(RES_DIR, f))))
new = pd.DataFrame(rows)
prev = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
allrows = pd.concat([prev, new], ignore_index=True)
allrows = allrows.drop_duplicates(subset=["scenario_id", "method"], keep="first")
allrows.to_csv(CSV, index=False)
print(f"MERGED: {len(allrows)} total rows -> {CSV}", flush=True)
