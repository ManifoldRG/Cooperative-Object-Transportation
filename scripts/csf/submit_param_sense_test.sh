#!/bin/bash --login
# ================================================================
# Sigma sweep — generates timestamped scenarios then submits array
# Run with: bash submit_sigma_sweep.sh
# ================================================================

REPO=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation
SCRATCH=/mnt/iusers01/eee01/r83771rr/scratch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENARIO_FILE=${SCRATCH}/results/scenarios_sigma_sweep_${TIMESTAMP}.json
OUTPUT_DIR=${SCRATCH}/results/sigma_sweep_${TIMESTAMP}

echo "Generating scenarios on login node..."
source ${REPO}/.venv/bin/activate
export PYTHONPATH=${REPO}:$PYTHONPATH
mkdir -p ${SCRATCH}/results ${SCRATCH}/logs

python - << PYEOF
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "${REPO}")
from spacecraft_libraries.evaluation.comparison import random_scenario_generator
log = []
for i in range(3):
    sys_p, bc, eps = random_scenario_generator(fixed_agents_num=3)
    sys_p.N = min(sys_p.N, 20)
    print(f"  Scenario {i+1}: N={sys_p.N}, n_agents={len(sys_p.rs)}, tf={bc.tf:.1f}s", flush=True)
    log.append({
        "scenario_id": i+1, "mu": sys_p.mu, "a": sys_p.a, "e": sys_p.e,
        "nu": sys_p.nu, "m": sys_p.m, "I_diag": np.diag(sys_p.I).tolist(),
        "N": sys_p.N, "tf": bc.tf, "epsilon": eps,
        "rs": [r.tolist() for r in sys_p.rs], "n_agents": len(sys_p.rs),
        "x0_r": bc.x0.r.tolist(), "x0_v": bc.x0.v.tolist(),
        "x0_phi": bc.x0.phi.tolist(), "x0_omega": bc.x0.omega.tolist(),
        "xf_r": bc.xf.r.tolist(), "xf_v": bc.xf.v.tolist(),
        "xf_phi": bc.xf.phi.tolist(), "xf_omega": bc.xf.omega.tolist(),
    })
json.dump(log, open("${SCENARIO_FILE}", "w"), indent=2)
print(f"Saved -> ${SCENARIO_FILE}", flush=True)
PYEOF

if [ ! -f ${SCENARIO_FILE} ]; then
    echo "ERROR: scenario file not created. Aborting."
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
echo "Scenarios: ${SCENARIO_FILE}"
echo "Output:    ${OUTPUT_DIR}"
echo "Submitting 30 tasks..."

sbatch << JOBEOF
#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --job-name=sigma_sweep
#SBATCH -a 1-30
#SBATCH -o ${SCRATCH}/logs/sigma_%A_%a.out
#SBATCH -e ${SCRATCH}/logs/sigma_%A_%a.err

echo "Task \${SLURM_ARRAY_TASK_ID}/30 started: \$(date)"
echo "Node: \$(hostname)"
echo "Scenario file: ${SCENARIO_FILE}"

module load python/3.13
source ${REPO}/.venv/bin/activate
export PYTHONPATH=${REPO}:\$PYTHONPATH

mkdir -p ${OUTPUT_DIR}/\${SLURM_ARRAY_JOB_ID}_results

python ${REPO}/scripts/csf/run_sensitivity.py \
    --task-id \${SLURM_ARRAY_TASK_ID} \
    --scenarios ${SCENARIO_FILE} \
    --output-dir ${OUTPUT_DIR}/\${SLURM_ARRAY_JOB_ID}_results

echo "Task \${SLURM_ARRAY_TASK_ID} finished: \$(date)"
JOBEOF

echo "Done. Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}"