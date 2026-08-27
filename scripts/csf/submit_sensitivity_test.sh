#!/bin/bash --login
# ================================================================
# run_sensitivity.py — OAT hyperparameter sweep, array-per-combo
#
# Run with: bash submit_run_sensitivity.sh
# ================================================================

REPO=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation
SCRATCH=/mnt/iusers01/eee01/r83771rr/scratch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=$(date +%s)
OUTPUT_DIR=${SCRATCH}/results/sensitivity_sweep_${TIMESTAMP}
SCENARIO_FILE=${OUTPUT_DIR}/scenarios_sensitivity_${TIMESTAMP}.json

# Number of Monte Carlo scenarios to generate. GD_SWEEPS in run_sensitivity.py
# has 3 params x 5 values, crossed with 2 methods (centralized_gd,
# decentralized_gd) -> 30 combos per scenario. N_TASKS tracks N_SCENARIOS
# automatically if you change it.
N_SCENARIOS=20
N_PARAMS=3
N_VALUES=5
N_METHODS=2
N_TASKS=$((N_SCENARIOS * N_PARAMS * N_VALUES * N_METHODS))

echo "Generating scenarios on login node..."
echo "Base seed: ${SEED} (scenario i -> seed ${SEED}+i)"
source ${REPO}/.venv/bin/activate
export PYTHONPATH=${REPO}:$PYTHONPATH
mkdir -p ${SCRATCH}/results ${SCRATCH}/logs ${OUTPUT_DIR}

python ${REPO}/scripts/csf/generate_scenarios.py \
    --n ${N_SCENARIOS} \
    --seed ${SEED} \
    --fixed-agents-num 6 \
    --output ${SCENARIO_FILE}

if [ ! -f ${SCENARIO_FILE} ]; then
    echo "ERROR: scenario file not created. Aborting."
    exit 1
fi

echo "Scenarios: ${SCENARIO_FILE}"
echo "Base seed: ${SEED}"
echo "Output:    ${OUTPUT_DIR}"
echo "Submitting ${N_TASKS} tasks (${N_SCENARIOS} scenarios x ${N_PARAMS} params x ${N_VALUES} values x ${N_METHODS} methods)..."

sbatch << JOBEOF
#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 7-0
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --job-name=run_sensitivity
#SBATCH -a 1-${N_TASKS}
#SBATCH -o ${SCRATCH}/logs/sensitivity_%A_%a.out
#SBATCH -e ${SCRATCH}/logs/sensitivity_%A_%a.err

echo "Task \${SLURM_ARRAY_TASK_ID}/${N_TASKS} started: \$(date)"
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