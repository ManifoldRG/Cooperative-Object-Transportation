#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH --job-name=baseline_comparison
#SBATCH -a 1-50%10
#SBATCH -o /mnt/iusers01/eee01/r83771rr/scratch/logs/baseline_%A_%a.out
#SBATCH -e /mnt/iusers01/eee01/r83771rr/scratch/logs/baseline_%A_%a.err

echo "Task ${SLURM_ARRAY_TASK_ID}/50 started: $(date)"
echo "Node: $(hostname)"

find /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation -name "__pycache__" -exec rm -rf {} + 2>/dev/null

source /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation/.venv/bin/activate
export PYTHONPATH=/mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation:$PYTHONPATH

cd /mnt/iusers01/eee01/r83771rr/scratch

python /mnt/iusers01/eee01/r83771rr/rev_mrgp/Cooperative-Object-Transportation/scripts/csf/baseline_comparison.py \
    --runs 5 \
    --time-limit 60 \
    --seed $((SLURM_ARRAY_TASK_ID * 1000)) \
    --output /mnt/iusers01/eee01/r83771rr/scratch/results/tasks/baseline_task_$(printf "%03d" ${SLURM_ARRAY_TASK_ID}).csv

echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"