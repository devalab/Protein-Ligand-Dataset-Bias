import torch 
import json,pickle,math
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


full_df = pd.read_csv(open('../kiba_all_pairs.csv','r'))

all_3_folds={}
for i in [0,1,2]:
    file_name = 'fold' +str(i)

    temp = open('../data/kiba/KIBA_3_FOLDS/' + file_name +'.pkl', 'rb')
    new_df = pd.read_pickle(temp)
    all_3_folds.update({file_name:new_df})
    temp.close()
        
def create_davis_test_train(test_fold_number,all_3_folds):
    
    test_set = pd.DataFrame(columns = full_df.columns)
    train_set = pd.DataFrame(columns= full_df.columns)
    for i in [0,1,2]:
        fold_name = 'fold' + str(i) 
        df = all_3_folds[fold_name]

        if str(i) == test_fold_number:
            test_set = df.copy()

        if str(i) != test_fold_number:
            train_set = pd.concat([train_set, df.copy()], ignore_index=True)

                
    return train_set, test_set

fold_number = '1'
train, test = create_davis_test_train(test_fold_number=fold_number, all_3_folds=all_3_folds)

test = test[['SMILES','Target Sequence','Label']]
train = train[['SMILES','Target Sequence','Label']]

# train =train.sample(100)
import rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem.Fingerprints import FingerprintMols
from Bio import pairwise2

train_targets = list(set(list(train['Target Sequence'])))
train_smiles = list(set(list(train['SMILES'])))

def computeLigandSimilarity(smiles):
    fingerprints = {}
    for smile in smiles:
        mol = AllChem.MolFromSmiles(smile)
        if mol == None:
            mol = AllChem.MolFromSmiles(smile, sanitize=False)
        fp = FingerprintMols.FingerprintMol(mol)
        fingerprints[smile] = fp
    
    n = len(smiles)
    sims = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            fpi = fingerprints[smiles[i]]
            fpj = fingerprints[smiles[j]]
            sim = fs(fpi, fpj)
            sims[i, j] = sims[j, i] = sim
    return sims

def computeProteinSimilarity(targets):
    n = len(targets)
    mat = np.zeros((n,n))
    mat_i = np.zeros(n)
    for i in range(n):
        seq = targets[i]
        s = pairwise2.align.localxx(seq,seq, score_only=True)
        mat_i[i] = s
        
    for i in range(n):
        print(i)
        for j in range(n):
            if mat[i][j] == 0 :
                s1 = targets[i]
                s2 = targets[j]
                sw_ij = pairwise2.align.localxx(s1,s2,score_only=True)
                normalized_score = sw_ij /math.sqrt(mat_i[i]*mat_i[j])
                mat[i][j] = mat[j][i] = normalized_score
    
    return mat

ligand_similarity_matrix = computeLigandSimilarity(train_smiles)
np.shape(ligand_similarity_matrix)

print(len(train_targets))
protein_similarity_matrix = computeProteinSimilarity(train_targets)


LSM = ligand_similarity_matrix
PSM = protein_similarity_matrix

test_targets = list(set(list(test['Target Sequence'])))
test_smiles = list(set(list(test['SMILES'])))

test_PSM = np.zeros((len(test_targets), len(train_targets)))
np.shape(test_PSM)


s_train_PSM = np.zeros(len(train_targets))
s_test_PSM = np.zeros(len(test_targets))

for i in range(len(train_targets)):
    seq = train_targets[i]
    s_train_PSM[i] = pairwise2.align.localxx(seq,seq, score_only=True)
    
for i in range(len(test_targets)):
    seq = test_targets[i]
    s_test_PSM[i] = pairwise2.align.localxx(seq,seq, score_only=True)
    
for i in range(len(test_targets)):
    print(i)
    for j in range(len(train_targets)):
        seq1 = test_targets[i]
        seq2 = train_targets[j]
        s_ij=pairwise2.align.localxx(seq1, seq2, score_only=True)
        N_S = s_ij / math.sqrt(s_train_PSM[j] * s_test_PSM[i])
        test_PSM[i][j] = N_S
        

test_LSM = np.zeros((len(test_smiles), len(train_smiles)))
np.shape(test_LSM)




test_smi_fp=[]
train_smi_fp = []

for smi in test_smiles:
    mol1 = AllChem.MolFromSmiles(smi)
    if mol1 == None:
        mol1= AllChem.MolFromSmiles(smi, sanitize=False)
    fp1 = FingerprintMols.FingerprintMol(mol1)
    test_smi_fp.append(fp1)

for smi in train_smiles:
    mol1 = AllChem.MolFromSmiles(smi)
    if mol1 == None:
        mol1= AllChem.MolFromSmiles(smi, sanitize=False)
    fp1 = FingerprintMols.FingerprintMol(mol1)
    train_smi_fp.append(fp1)
    
    
for i in range(len(test_smiles)):
#     print(i)
    for j in range(len(train_smiles)):
        smi1 = test_smiles[i]
        smi2 = train_smiles[j]
        
        test_LSM[i][j] = fs(test_smi_fp[i], train_smi_fp[j])
        
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 20
# num_classes = 10
batch_size = 18
learning_rate = 0.001


class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, smiles,smi_index ,targets, target_index, LSM,PSM,transform=None):
        self.df = dataframe
#         self.root_dir = root_dir
        self.smiles =smiles
        self.targets = targets
        self.LSM = LSM
        self.PSM = PSM
        self.transform = transform
        self.smi_index=smi_index
        self.target_index=target_index
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        smi = self.df.iloc[idx]['SMILES']
        seq = self.df.iloc[idx]['Target Sequence']
#         s_i = self.smiles.index(smi)
#         t_i = self.targets.index(seq)
        s_i = self.smi_index[smi]
        t_i = self.target_index[seq]
        
        ki=self.LSM[s_i]
        kj=self.PSM[t_i]
        
        ki_x_kj = np.outer(ki,kj)
        ki_x_kj = torch.tensor([ki_x_kj])
        output = {'outer_product': ki_x_kj , 'Label':self.df.iloc[idx]['Label']}
        return output
    
test_index_smi={}
for i in test_smiles:
    test_index_smi[i]=test_smiles.index(i)
train_index_smi={}
for i in train_smiles:
    train_index_smi[i]=train_smiles.index(i)
    

test_index_seq={}
for i in test_targets:
    test_index_seq[i]=test_targets.index(i)

train_index_seq={}
for i in train_targets:
    train_index_seq[i]=train_targets.index(i)
    
train_dataset = custom_dataset(dataframe=train, smiles=train_smiles, smi_index=train_index_smi, targets = train_targets, target_index=train_index_seq, LSM=LSM,PSM=PSM)


test_dataset = custom_dataset(dataframe=test, smiles=test_smiles,smi_index=test_index_smi, targets = test_targets, target_index=test_index_seq, LSM=test_LSM,PSM=test_PSM)


train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(len(train_loader)*batch_size +  len(test_loader)*batch_size)
        
        
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32, 5).double()
        self.pool1 = nn.MaxPool2d(2,2).double()
        self.conv2 = nn.Conv2d(32,18,3).double()
        self.pool2 = nn.MaxPool2d(2,2).double()
        self.fc1 = nn.Linear(18*514*37, 128).double()
        self.fc2 = nn.Linear(128,1).double()
        self.dropout = nn.Dropout(0.1).double()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,18*514*37)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        return x
    
    
model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def rmse(y,f):
    rmse = math.sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
from lifelines.utils import concordance_index
def ci(y,f):
    return concordance_index(y,f)

def predicting(model, device, test_loader):
    model.eval()
    total_preds = np.array([])
    total_labels = np.array([])
    with torch.no_grad():
        correct = 0
        total = 0
        c=0
        for i in test_loader:
            print("in testing ",c, flush=True)
            c=c+1
            images = i['outer_product']
            labels = i['Label']
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images) 
            outputs = outputs.cpu().detach().numpy().flatten()
            labels =labels.cpu().detach().numpy().flatten()
            total_preds = np.concatenate([total_preds, outputs])
            total_labels = np.concatenate([total_labels, labels])
    model.train()   
    return total_labels, total_preds

model_file_name = 'best_sim-CNN-DTA_kiba_fold' + fold_number + "NEW"+ '.model'
result_file_name = 'best_result_sim-CNNDTA_kiba_fold'+fold_number + "NEW"+ '.csv'

# Train the model
best_mse = 1000
best_ci = 0

total_step = len(train_loader)
for epoch in range(num_epochs):
    c=0
    for i in train_loader:
        c=c+1
        images = i['outer_product']
        labels = i['Label']
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.flatten(), labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
               .format(epoch+1, num_epochs, c, total_step, loss.item()),flush=True)
    
#     taking best model so far
    G,P = predicting(model, device, test_loader)
    ret = [rmse(G, P), mse(G, P)]
    print(ret,flush=True)
    if ret[1] < best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name, 'w') as f:
            f.write(','.join(map(str, ret)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_ci = 0
        best_r = 0
        
        print('rmse improved at epoch ', best_epoch,
                      '; best_mse,best_ci,best_r:', best_mse, best_ci,best_r, flush=True)
        
        
model = ConvNet().to(device)
model.load_state_dict(torch.load('./best_sim-CNN-DTA_kiba_fold0NEW.model'))
model.eval()
total_preds = np.array([])
total_labels = np.array([])
with torch.no_grad():
    correct = 0
    total = 0
    for i in test_loader:
        images = i['outer_product']
        labels = i['Label']
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images) 
        outputs = outputs.cpu().detach().numpy().flatten()
        labels =labels.cpu().detach().numpy().flatten()
        total_preds = np.concatenate([total_preds, outputs])
        total_labels = np.concatenate([total_labels, labels])
#         total_preds = torch.cat(total_preds, outputs.cpu(), 0 )
#         total_labels = torch.cat(total_labels, labels.cpu(), 0)
#         break

G,P = total_labels, total_preds

print("MSE = ",mse(G,P),flush=True)
print("R = ",pearson(G,P),flush=True)
print("CI = ",ci(G,P),flush=True)
print("RMSE = ",rmse(G,P),flush=True)








