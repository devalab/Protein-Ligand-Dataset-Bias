#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_kiba_fold1_NEW_NEW.txt

echo "starting"
PYTHONUNBUFFERED=1
python3 new_kiba_fold1.py

echo "finished"


