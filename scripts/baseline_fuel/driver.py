"""FUEL-objective baseline campaign driver: {D-GD, C-GD, NLP cold} x
{10, 30, 60, 300 s} x first-10-scenarios-per-angle at 1x timesteps, every
returned U independently rollout-verified. One subprocess per task (avoids
CasADi worker leaks/teardown segfaults); resume-safe via per-task JSONs;
hung tasks killed at budget-scaled caps and recorded as ABSTAIN:DriverTimeout.

Run:  python driver.py            (env: WORKERS=5, BUDGETS=10,30,60,300)
Out:  results/fuel_baseline_1xT.csv (merged at the end of every run)
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results_json")
OUT_CSV = os.path.join(HERE, "results", "fuel_baseline_1xT.csv")
WORKER = os.path.join(HERE, "task_worker.py")
K = int(os.environ.get("WORKERS", "5"))
BUDGETS = [float(b) for b in os.environ.get("BUDGETS", "10,30,60,300").split(",")]
T_MULT = 1
METHODS = ["decentralized_gd", "centralized_gd", "centralized_nlp_th"]
REPO = os.environ.get("COT_REPO") or os.path.dirname(os.path.dirname(HERE))

import pandas as pd

sc = pd.read_csv(os.path.join(REPO, "CSF Runs", "results_20260823",
                              "baseline_large_20_scenarios_th.csv"))
sc = sc.groupby("thrust_angle_deg", group_keys=False).head(10)

# ascending budgets: the cheap sweeps land complete first
wanted = [(int(r.global_scenario_id), m, b)
          for b in sorted(BUDGETS)
          for _, r in sc.iterrows()
          for m in METHODS]

os.makedirs(RES_DIR, exist_ok=True)
done = {f[:-5] for f in os.listdir(RES_DIR) if f.endswith(".json")}
todo = [t for t in wanted if f"{t[0]}_{t[1]}_{int(t[2])}" not in done]
print(f"{len(done)} done, {len(todo)} to run (workers={K}, budgets={BUDGETS})",
      flush=True)


def cap_s(method, budget):
    # dGD budget is per-agent (6x wall) + build/solve overhead; fuel cold
    # solves can run minutes past the budget check
    if method == "decentralized_gd":
        return 6 * budget + 900
    return budget + 600


running = {}  # popen -> (sid, method, budget, t0)
t_start = time.perf_counter()
finished = 0
total = len(todo)
while todo or running:
    while todo and len(running) < K:
        sid, m, b = todo.pop(0)
        p = subprocess.Popen([sys.executable, WORKER, str(sid), m, str(b),
                              str(T_MULT), RES_DIR],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running[p] = (sid, m, b, time.perf_counter())
    time.sleep(5)
    for p in list(running):
        sid, m, b, t0 = running[p]
        age = time.perf_counter() - t0
        if p.poll() is None and age > cap_s(m, b):
            p.kill()
            p.wait()
            row_sc = sc[sc.global_scenario_id == sid].iloc[0]
            with open(os.path.join(RES_DIR, f"{sid}_{m}_{int(b)}.json"), "w") as f:
                json.dump(dict(scenario_id=sid,
                               thrust_angle_deg=float(row_sc.thrust_angle_deg),
                               method=m, n_steps=int(row_sc.N) * T_MULT,
                               time_limit_s=b, wall_s=round(age, 1),
                               cost_reported=None, cost_rollout=None,
                               V_rollout=None,
                               ipopt_status="ABSTAIN:DriverTimeout",
                               converged=""), f)
        if p.poll() is not None:
            running.pop(p)
            finished += 1
            have = os.path.exists(os.path.join(RES_DIR, f"{sid}_{m}_{int(b)}.json"))
            print(f"[{finished}/{total}] {sid} {m} {int(b)}s exit={p.returncode} "
                  f"json={'ok' if have else 'MISSING'} wall={age/60:.1f}min "
                  f"({(time.perf_counter()-t_start)/60:.1f} min)", flush=True)

# merge all JSONs into one CSV
rows = [json.load(open(os.path.join(RES_DIR, f)))
        for f in sorted(os.listdir(RES_DIR)) if f.endswith(".json")]
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).sort_values(
    ["time_limit_s", "thrust_angle_deg", "scenario_id", "method"]).to_csv(
    OUT_CSV, index=False)
print(f"MERGED: {len(rows)} rows -> {OUT_CSV}", flush=True)
