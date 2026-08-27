#!/bin/bash --login

# Partition: multicore:  2-168 CPUs per job; multicore_small : 2-31 CPUS; serial : 1 CPU per job
#SBATCH -p serial
# Time limit: HH:MM:SS max: 7-0 (7 days).
#SBATCH -t 7-0
# CPU Cores
#SBATCH -n 1
# Memory: went with 8G just in case
#SBATCH --mem=2G
# Job name
#SBATCH --job-name=dcgd_scalability_ol
# Job array: -a 1-N    : run tasks numbered 1 to N (1-indexed)
# note: only 10 tasks at a time are accepted for serial partition and 42 for multicore, but its queued, meaning the first 42 tasks are run and then the next 42 tasks are run. SLURm automatically does this so its ok to leave it beyond the task limit
#
# Full grid, one array job, no manual resubmission:
#   4 agent counts x 6 thrust angles x 20 tasks-per-(agent,angle) = 480.
# #SBATCH directives can't reference shell variables, so this bound is
# hardcoded — if AGENTS_LIST/THRUST_ANGLES_DEG below change length, update
# this too (N_TASKS = len(AGENTS_LIST) * len(THRUST_ANGLES_DEG) * 20).
#SBATCH -a 1-480

# Output files
# %A = job array ID (same for all tasks)
# %a = individual task ID (unique per task)
# currently have setup a logs/ folder to track any issues:
#SBATCH -o /mnt/iusers01/eee01/r83771rr/scratch/logs/scalability_ol_%A_%a.out
#SBATCH -e /mnt/iusers01/eee01/r83771rr/scratch/logs/scalability_ol_%A_%a.err


# Job body

# Agent-count grid (was 4 manual resubmissions with --fixed-agents-num
# hand-edited each time; now one array covers all four).
AGENTS_LIST=(4 11 18 25)

# Thrust-angle cone ablation.
THRUST_ANGLES_DEG=(1 2 2.5 5 15 60)

TASKS_PER_COMBO=20
N_ANGLES=${#THRUST_ANGLES_DEG[@]}
TASKS_PER_AGENT=$(( N_ANGLES * TASKS_PER_COMBO ))   # 6 * 20 = 480

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))                  # 0-indexed
AGENT_IDX=$(( IDX / TASKS_PER_AGENT ))
REM=$(( IDX % TASKS_PER_AGENT ))
ANGLE_IDX=$(( REM / TASKS_PER_COMBO ))
LOCAL_TASK_ID=$(( REM % TASKS_PER_COMBO + 1 ))

FIXED_AGENTS=${AGENTS_LIST[$AGENT_IDX]}
THRUST_ANGLE_DEG=${THRUST_ANGLES_DEG[$ANGLE_IDX]}
THRUST_ANGLE_RAD=$(awk "BEGIN{print ${THRUST_ANGLE_DEG} * 3.14159265358979 / 180}")

echo "Task ${SLURM_ARRAY_TASK_ID}/480 started: $(date)"
echo "Node: $(hostname)"
echo "Agents: ${FIXED_AGENTS}  Angle: ${THRUST_ANGLE_DEG} deg (${THRUST_ANGLE_RAD} rad)  local_task_id=${LOCAL_TASK_ID}"
SEED=${SLURM_ARRAY_TASK_ID}

# Activate Python venv
source /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation/.venv/bin/activate

# Add repo to Python path
export PYTHONPATH=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation:$PYTHONPATH

# Change to scratch directory
cd /mnt/iusers01/eee01/r83771rr/scratch

# Separate output folder per agent count, e.g. agents_11/, so the 4 configs
# never mix in one directory.
OUTPUT_DIR=/mnt/iusers01/eee01/r83771rr/scratch/scalability_openloop/agents_${FIXED_AGENTS}/
mkdir -p ${OUTPUT_DIR}

python  /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation/scripts/csf/run_mc_scalability_dgd_csf.py \
     --task-id ${SLURM_ARRAY_TASK_ID} \
     --output-dir ${OUTPUT_DIR} \
     --fixed-agents-num ${FIXED_AGENTS} \
     --no-fault \
     --thrust_angle ${THRUST_ANGLE_RAD} \
     --seed "$SEED"


echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"