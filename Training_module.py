# Input to this code is the basic Co-Occurence Matrix ( alias - co_occ_copy)
#GloVe Basic code- 

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE

class GloveDataset:
    
    def __init__(self,data):

        self._i_idx=list()
        self._j_idx=list()
        self._Xij=list()

        for ii,jj in data.items():
            for kk,ll in jj.items():
                self._i_idx.append(ii)
                self._j_idx.append(kk)
                self._Xij.append(ll)
        
        self._i_idx=[float(i) for i in self._i_idx]
        self._j_idx=[float(i) for i in self._j_idx]
        self._Xij=[float(i) for i in self._Xij]
        
        
        self._i_idx=torch.LongTensor(self._i_idx)
        self._j_idx=torch.LongTensor(self._j_idx)
        self._Xij=torch.LongTensor(self._Xij)
    
    def batch_split(self,batch_size):
        rand_idx=torch.LongTensor(np.random.choice(len(self._Xij),len(self._Xij),replace=False))
        
        for p in range(0,len(rand_idx),batch_size):
            batch_ids=rand_idx[p:p+batch_size]
            yield self._Xij[batch_ids],self._i_idx[batch_ids],self._j_idx[batch_ids]
            
data=GloveDataset(co_occ_copy)

#co_occ_copy - Basic Co-Occurence matrix which is used as a input here

class GloveModel(nn.Module):
    
    def __init__(self,num_embeddings,embed_dim):
        super(GloveModel, self).__init__()
        
        self.wi=nn.Embedding(num_embeddings,embed_dim)
        self.wj=nn.Embedding(num_embeddings,embed_dim)
        self.bi=nn.Embedding(num_embeddings,1)
        self.bj=nn.Embedding(num_embeddings,1)
        
        self.wi.weight.data.uniform_(-1,1)
        self.wj.weight.data.uniform_(-1,1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
        
    def forward(self,i_indices,j_indices):
        w_i=self.wi(i_indices)
        w_j=self.wj(j_indices)
        b_i=self.bi(i_indices).squeeze()
        b_j=self.bj(j_indices).squeeze()
            
        x=torch.sum(w_i*w_j,dim=1) + b_i + b_j
            
        return x
        
 glove=GloveModel(len(vocab),50)
 
 #Vocab- total number of words in our corpus. It helps in initializing out Embedding Vector
 
 # Various function defined for purpose

def weight_func(x,x_max,alpha):
    wx=(x.to(dtype=torch.float)/float(x_max))**float(alpha)
    wx=torch.min(wx,torch.ones_like(wx))
    return wx

def wsme_loss(weights,inputs,targets):
    loss=weights*F.mse_loss(inputs,targets,reduction='none')
    return torch.mean(loss)

optimizer = optim.Adagrad(glove.parameters(), lr=0.05)

import math

n_epochs=100
batch_size=2000
x_max=100
alpha=0.75

n_batches=int(len(data._Xij)/batch_size)

print(n_batches)

loss_values=list()

for e in range(1,n_epochs+1):
    
    batch_i=0
    
    for x_ij,i_idx,j_idx in data.batch_split(batch_size):
        
        batch_i += 1
        optimizer.zero_grad()
        
        outputs=glove(i_idx,j_idx)
        weights_x=weight_func(x_ij,x_max,alpha)
        loss=wsme_loss(weights_x.float(),outputs.float(),torch.log(x_ij.float()))
        
        loss.backward()
        
        optimizer.step()
        
        loss_values.append(loss.item())
#         print(loss.item())
        
        print ("Epoch: {}/{} \t Batch: {}/{} \t Loss:{}".format (e,n_epochs,batch_i,n_batches,
                                                                    np.mean(loss_values[-20:])))
        
print ("GloVe is Completed. Time to Chillex")

print (len(i_idx),len(j_idx),len(x_ij))

%matplotlib inline

#To plot the loss values from the training process
plt.plot(loss_values)

# Saving all the embeddings - To be utilized further
emb_i = glove.wi.weight.cpu().data.numpy()
emb_j = glove.wj.weight.cpu().data.numpy()

emb=emb_i+emb_j

#Visualizing the Embeddings which are generated

top_k = 350

tsne = TSNE(metric='cosine', random_state=123,perplexity=12)
embed_tsne = tsne.fit_transform(emb[:top_k, :])

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(top_k):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(id2offer[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
