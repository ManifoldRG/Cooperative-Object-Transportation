#!/bin/bash --login
# Partition: multicore:  2-168 CPUs per job; multicore_small : 2-31 CPUS; serial : 1 CPU per job
#SBATCH -p serial
# Time limit: HH:MM:SS max: 7-0 (7 days).
#SBATCH -t 1:00:00
# CPU Cores
#SBATCH -n 1
# Memory: went with 8G just in case
#SBATCH --mem=2G
# Job name
#SBATCH --job-name=mppi_sens_mc
# Job array: -a 1-N    : run tasks numbered 1 to N (1-indexed)
#SBATCH -a 1-162

# Output files
# %A = job array ID (same for all tasks)
# %a = individual task ID (unique per task)
# currently have setup a logs/ folder to track any issues:
#SBATCH -o ~/scratch/logs/sens_mc_%A_%a.out
#SBATCH -e ~/scratch/logs/sens_mc_%A_%a.err


# Job body

echo "Task ${SLURM_ARRAY_TASK_ID}/162 started: $(date)"
echo "Node: $(hostname)"

# Activate Python venv
source ~/rev_mrgp/Cooperative-Object-Transportation/.venv/bin/activate

# Add repo to Python path
export PYTHONPATH=~/rev_mrgp/Cooperative-Object-Transportation:$PYTHONPATH

# Change to scratch directory
cd ~/scratch

# Run the task basically running the code and reading the scenarios to be run from the mrgp folder, and writing output to scratch folder
python ~/rev_mrgp/Cooperative-Object-Transportation/scripts/csf/run_sensitivity_task.py \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --scenarios ~/rev_mrgp/Cooperative-Object-Transportation/scripts/csf/results/scenarios.json \
    --output-dir ~/scratch/results/tasks

echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"