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
#SBATCH --job-name=mppi_scalability_ol
# Job array: -a 1-N    : run tasks numbered 1 to N (1-indexed)
# note: only 10 tasks at a time are accepted for serial partition and 42 for multicore, but its queued, meaning the first 42 tasks are run and then the next 42 tasks are run. SLURm automatically does this so its ok to leave it beyond the task limit
#SBATCH -a 1-162

# Output files
# %A = job array ID (same for all tasks)
# %a = individual task ID (unique per task)
# currently have setup a logs/ folder to track any issues:
#SBATCH -o /mnt/iusers01/eee01/r83771rr/scratch/logs/thrustangle_ol_%A_%a.out
#SBATCH -e /mnt/iusers01/eee01/r83771rr/scratch/logs/thrustangle_ol_%A_%a.err


# Job body

# mkdir -p /mnt/iusers01/eee01/r83771rr/scratch/\${TIMESTAMP}_results

echo "Task ${SLURM_ARRAY_TASK_ID}/162 started: $(date)"
echo "Node: $(hostname)"

# Activate Python venv
source /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation/.venv/bin/activate

# Add repo to Python path
export PYTHONPATH=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation:$PYTHONPATH

# Change to scratch directory
cd /mnt/iusers01/eee01/r83771rr/scratch

# define thrust angle here by using awk. modify pi / num
pi=3.141592653589793
thrust_angle=$(awk -v pi="$pi" 'BEGIN { print pi / 16 }')

# thust angle ablation run example with 6 agents ( repeat with pi/2, pi/4, pi/8, pi/16)
python  /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation/scripts/csf/run_montecarlo_scalability_test_csf.py \
     --task-id ${SLURM_ARRAY_TASK_ID} \
     --output-dir /mnt/iusers01/eee01/r83771rr/scratch/scalability_thrust_angle_ablation/ \
     --fixed-agents-num 6 \
     --thrust_angle "$thrust_angle" \
     --no-fault


echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"