#!/bin/bash --login
# ================================================================
# Baseline comparison sweep — generates scenarios then submits array job.
# Run with: bash submit_baseline.sh   (NOT sbatch — this script itself runs
# on the login node to generate scenarios first, then submits the array).
# ================================================================

REPO=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation
SCRATCH=/mnt/iusers01/eee01/r83771rr/scratch

# decentralized_mppi scales its time_limit internally by num_agents (each
# agent gets the full per-task time_limit). GA's deadline check is
# per-generation (centralized_ga.py:70-71), so it can overshoot time_limit
# by up to one generation's runtime — run the overshoot check before
# trusting the sweep.

# Number of Monte Carlo scenarios to generate. Same 20 scenarios are reused
# across all 4 time limits (60/300/600/1200s) and all 9 methods (cold NLP,
# warm NLP, GA, centralized/decentralized GS, centralized/decentralized GD,
# centralized/decentralized MPPI).
N_SCENARIOS=10

N_METHODS=9
N_TIME_LIMITS=4
N_TASKS=$((N_SCENARIOS * N_METHODS * N_TIME_LIMITS))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=$(date +%s)
OUTPUT_DIR=${SCRATCH}/results/baseline_${TIMESTAMP}
SCENARIO_FILE=${SCRATCH}/results/baseline_${TIMESTAMP}/scenarios_baseline_${TIMESTAMP}.json


echo "Generating ${N_SCENARIOS} scenarios on login node..."
echo "Base seed: ${SEED}"
source ${REPO}/.venv/bin/activate
export PYTHONPATH=${REPO}:$PYTHONPATH
mkdir -p ${SCRATCH}/results ${SCRATCH}/logs

python ${REPO}/scripts/csf/generate_scenarios.py \
    --n ${N_SCENARIOS} \
    --seed ${SEED} \
    --fixed-agents-num 6 \
    --output ${SCENARIO_FILE}

if [ ! -f ${SCENARIO_FILE} ]; then
    echo "ERROR: scenario file not created. Aborting."
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
echo "Scenarios:  ${SCENARIO_FILE}"
echo "Base seed:  ${SEED}"
echo "Output:     ${OUTPUT_DIR}"
echo "Submitting ${N_TASKS} tasks (${N_SCENARIOS} scenarios x ${N_METHODS} methods x ${N_TIME_LIMITS} time limits)..."

sbatch << JOBEOF
#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 7-0
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --job-name=baseline_comparison
#SBATCH -a 1-${N_TASKS}
#SBATCH -o ${SCRATCH}/logs/baseline_%A_%a.out
#SBATCH -e ${SCRATCH}/logs/baseline_%A_%a.err

echo "Task \${SLURM_ARRAY_TASK_ID}/${N_TASKS} started: \$(date)"
echo "Node: \$(hostname)"
echo "Scenario file: ${SCENARIO_FILE}"
echo "Base seed: ${SEED}"

find ${REPO} -name "__pycache__" -exec rm -rf {} + 2>/dev/null

source ${REPO}/.venv/bin/activate
export PYTHONPATH=${REPO}:\$PYTHONPATH

python ${REPO}/scripts/csf/baseline_comparison.py \
    --task-id \${SLURM_ARRAY_TASK_ID} \
    --scenarios ${SCENARIO_FILE} \
    --output-dir ${OUTPUT_DIR}

echo "Task \${SLURM_ARRAY_TASK_ID} finished: \$(date)"
JOBEOF

echo "Done. Monitor: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}"