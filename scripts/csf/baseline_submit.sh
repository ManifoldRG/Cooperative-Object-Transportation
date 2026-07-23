#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --job-name=baseline_comparison
#SBATCH -a 1-50%10
#SBATCH -o ~/scratch/logs/baseline_%A_%a.out
#SBATCH -e ~/scratch/logs/baseline_%A_%a.err

echo "Task ${SLURM_ARRAY_TASK_ID}/50 started: $(date)"
echo "Node: $(hostname)"

find ~/rev_mrgp/Cooperative-Object-Transportation -name "__pycache__" -exec rm -rf {} + 2>/dev/null

source ~/rev_mrgp/Cooperative-Object-Transportation/.venv/bin/activate
export PYTHONPATH=~/rev_mrgp/Cooperative-Object-Transportation:$PYTHONPATH

cd ~/scratch

python ~/rev_mrgp/Cooperative-Object-Transportation/scripts/csf/baseline_comparison.py \
    --runs 5 \
    --time-limit 60 \
    --seed $((SLURM_ARRAY_TASK_ID * 1000)) \
    --output ~/scratch/results/tasks/baseline_task_$(printf "%03d" ${SLURM_ARRAY_TASK_ID}).csv

echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"