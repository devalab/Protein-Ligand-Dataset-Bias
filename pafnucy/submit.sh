#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_fold0-ligand-only.txt

PYTHONUNBUFFERED=1

# python3 training.py --input_dir HDF/ --output_prefix ./PUBLISHED_RESULTS/output --batch_size=5
python3 training.py --input_dir SPLITS/LIGAND_ONLY_DATA/ --output_prefix ./LIGAND_ONLY_FOLD0/output --batch_size=5 

