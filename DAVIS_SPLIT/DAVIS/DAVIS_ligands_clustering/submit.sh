#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=davis_l-0.9_ligand_split_using_gnina_clustering.txt

echo "starting"

python3 MY_CLUSTERING.py --pdbfiles pdb_files -i INPUT -o OUTPUT_seed0 -c OUTPUT_seed0 -n 3 -s 1 -s2 0 -l 0.9 --randomize 0 -v

echo "python3 MY_CLUSTERING.py --pdbfiles pdb_files -i INPUT -o OUTPUT_seed0 -c OUTPUT_seed0 -n 3 -s 1 -s2 0 -l 0.9 --randomize 0 -v
 " 

echo "finished"

