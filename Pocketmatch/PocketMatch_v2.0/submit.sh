#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_PocketMatch_final_results.txt

PYTHONUNBUFFERED=1

./Step3-PM_serial alpha-file_maker/outfile.cabbage
