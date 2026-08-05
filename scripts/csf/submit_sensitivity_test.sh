#!/bin/bash --login
# ================================================================
# run_sensitivity.py — OAT hyperparameter sweep, array-per-combo
#
# Login node generates 5 scenarios (fixed-agents-num=6), then submits an
# array of 200 tasks: 5 scenarios x 4 params x 5 values x 2 methods.
# Each task runs exactly one combo and writes its own task_NNNN.csv.
#
# Replaces run_oat_task.py + submit_sensitivity_test.sh — run_sensitivity.py
# now carries the same BASELINE/SWEEPS/build_combos/--task-id design.
#
# Run with: bash submit_run_sensitivity.sh
# ================================================================

REPO=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation
SCRATCH=/mnt/iusers01/eee01/r83771rr/scratch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=$(date +%s)
SCENARIO_FILE=${SCRATCH}/results/scenarios_sensitivity_${TIMESTAMP}.json
OUTPUT_DIR=${SCRATCH}/results/sensitivity_sweep_${TIMESTAMP}

echo "Generating scenarios on login node..."
echo "Base seed: ${SEED} (scenario i -> seed ${SEED}+i)"
source ${REPO}/.venv/bin/activate
export PYTHONPATH=${REPO}:$PYTHONPATH
mkdir -p ${SCRATCH}/results ${SCRATCH}/logs

python ${REPO}/scripts/csf/generate_scenarios.py \
    --n 5 \
    --seed ${SEED} \
    --fixed-agents-num 6 \
    --output ${SCENARIO_FILE}

if [ ! -f ${SCENARIO_FILE} ]; then
    echo "ERROR: scenario file not created. Aborting."
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
echo "Scenarios: ${SCENARIO_FILE}"
echo "Base seed: ${SEED}"
echo "Output:    ${OUTPUT_DIR}"
echo "Submitting 200 tasks..."

sbatch << JOBEOF
#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --job-name=run_sensitivity
#SBATCH -a 1-200
#SBATCH -o ${SCRATCH}/logs/sensitivity_%A_%a.out
#SBATCH -e ${SCRATCH}/logs/sensitivity_%A_%a.err

echo "Task \${SLURM_ARRAY_TASK_ID}/200 started: \$(date)"
echo "Node: \$(hostname)"
echo "Scenario file: ${SCENARIO_FILE}"
echo "Base seed: ${SEED}"

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