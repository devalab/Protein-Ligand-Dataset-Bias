import os
import oddt
import numpy as np
import pandas as pd
import  tqdm

from oddt.fingerprints import InteractionFingerprint, PLEC, SPLIF

paths = {}

path_to_general = './PDBBIND/v2019-other-PL/'
for pdbid in os.listdir(path_to_general):
    if pdbid == 'index' or pdbid == 'readme':
        continue
    ligand_path = path_to_general + pdbid + '/' + pdbid + '_ligand.sdf'
    protein_path = path_to_general + pdbid + '/' + pdbid + '_protein.pdb'
    paths[pdbid] = {}
    paths[pdbid]['ligand_path'] = ligand_path
    paths[pdbid]['protein_path'] = protein_path
    

path_to_general = './PDBBIND/refined-set/'
for pdbid in os.listdir(path_to_general):
    if pdbid == 'index' or pdbid == 'readme':
        continue
    ligand_path = path_to_general + pdbid + '/' + pdbid + '_ligand.sdf'
    protein_path = path_to_general + pdbid + '/' + pdbid + '_protein.pdb'
    paths[pdbid] = {}
    paths[pdbid]['ligand_path'] = ligand_path
    paths[pdbid]['protein_path'] = protein_path
    
    
path_to_general = './PDBBIND/CORE-FULL/CASF-2016/coreset/'
for pdbid in os.listdir(path_to_general):
    if pdbid == 'index' or pdbid == 'readme':
        continue
    ligand_path = path_to_general + pdbid + '/' + pdbid + '_ligand.sdf'
    protein_path = path_to_general + pdbid + '/' + pdbid + '_protein.pdb'
    paths[pdbid] = {}
    paths[pdbid]['ligand_path'] = ligand_path
    paths[pdbid]['protein_path'] = protein_path
    
    
path_to_general = './PDBBIND/CORE-FULL/CASF-2013/coreset/'
for pdbid in os.listdir(path_to_general):
    if pdbid == 'index' or pdbid == 'readme' or pdbid == 'README':
        continue
    ligand_path = path_to_general + pdbid + '/' + pdbid + '_ligand.sdf'
    protein_path = path_to_general + pdbid + '/' + pdbid + '_protein.pdb'
    paths[pdbid] = {}
    paths[pdbid]['ligand_path'] = ligand_path
    paths[pdbid]['protein_path'] = protein_path
    
path_to_general = './PDBBIND/CORE-FULL/CASF/protein/pdb/'
for pdbid in os.listdir(path_to_general):
    if pdbid == 'index' or pdbid == 'readme' or pdbid == 'README':
        continue
    ligand_path = './PDBBIND/CORE-FULL/CASF/ligand/ranking_scoring/crystal_sdf/' + pdbid[:4] + '_ligand.sdf'
    protein_path = path_to_general + pdbid 
    paths[pdbid] = {}
    paths[pdbid]['ligand_path'] = ligand_path
    paths[pdbid]['protein_path'] = protein_path
   
def create_fingerprint(key):
    ligand = next(oddt.toolkit.readfile('sdf', paths[key]['ligand_path']))
    protein = next(oddt.toolkit.readfile('pdb',paths[key]['protein_path']))
    fp = PLEC(ligand, protein,sparse=True,count_bits=True, size=16384)
    return [key,fp]

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
fingerprints_list = Parallel(n_jobs=num_cores)(delayed(create_fingerprint)(key) for key in list(paths.keys()))

import pickle

with open('fingerprints_list_16k.pkl', 'wb') as f:
    pickle.dump(fingerprints_list, f)

