
import os 

path = './PDBBIND/v2019-other-PL/'


ls = []
for dirname in os.listdir(path):
    if dirname == 'readme' or dirname == 'index':
        continue
    
    cur_path = path + '/' + dirname
    protein_path = cur_path + '/' + dirname + '_protein.pdb'
    ligand_path = cur_path + '/' + dirname + '_ligand.sdf'
#     protein_path = './CASF/protein/mol2/' + dirname + '_protein.mol2'
#     ligand_path = './CASF/ligand/ranking_scoring/crystal_mol2/' + dirname + '_ligand.mol2'
    
    cmd = 'smina ' + ' -r ' + protein_path +  ' -l ' + ligand_path + ' --score_only |' + 'grep Affinity'
    stream = os.popen(cmd)
    ls.append(dirname + ' ' + stream.read())
    print(len(ls), ls[-1], flush=True)

f = open('general-2019-Affinity.csv', 'w')
for i in ls:
    f.write(i)
f.close()
