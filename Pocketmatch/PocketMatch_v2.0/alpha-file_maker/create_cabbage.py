import os 

i= 0 
for pdb in os.listdir('4_DIST'):
    cmd = 'bash Step0-cabbage.sh 4_DIST/' + pdb + ' -type1'
    print(i, cmd)
    i += 1
    os.system(cmd)
