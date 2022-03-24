import rdkit
import rdkit.Chem
from rdkit import Chem
import rdkit.Chem.Descriptors
from rdkit.Chem.Descriptors import ExactMolWt,HeavyAtomMolWt
from clean_pdb import clean_pdb
import os
import sys
count=0
count_prot=0
f=open('exclude_structs_'  + sys.argv[1] + '.txt','w')
for dir in sorted(os.listdir('/scratch/kanakala.ganesh/2019/' + sys.argv[1])):
    if len(dir)!=4:
        continue
    file=dir+'_protein.pdb'
    mol = Chem.MolFromMolFile(
        os.path.join('/scratch/kanakala.ganesh/2019/'+ sys.argv[1], dir, file.replace('protein.pdb', 'ligand.sdf')), sanitize=False)
    if mol is None:
        f.write(dir + '\n')
        count += 1
        continue
    mol_wt = HeavyAtomMolWt(mol)
    if mol_wt > 1000:
        f.write(dir + '\n')
        count += 1
        continue
    try:
        clean_pdb(os.path.join('/scratch/kanakala.ganesh/2019/'+ sys.argv[1], dir, file),
                  os.path.join('/scratch/kanakala.ganesh/2019/'+ sys.argv[1], dir, file.replace('protein', 'protein_nowat')))
    except:
        f.write(dir+'\n')
        count+=1
        continue

print(count)