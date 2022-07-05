import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from blocks import *
 
class DANet(nn.Module):
    def __init__(self,dataset):
        super(DAGCN,self).__init__()
        options = {'seed':[62,3,5],'seed iv':[62,4,5]}
        self.chan_num = options[dataset][0]
        self.class_num = options[dataset][1]
        self.band_num = options[dataset][2]

        self.gcn_DE = HGCN(dim = 1, chan_num = self.chan_num,band_num=self.band_num, reduction_ratio = 128, si = 256)
        self.gcn_PSD = HGCN(dim = 1, chan_num = self.chan_num,band_num=self.band_num, reduction_ratio = 128, si = 256)
        self.ATFFNet = Encoder(n_layers=2,n_heads=5,d_model=self.band_num*2,d_k=8,d_v=8,d_ff=10)
        
        self.A = torch.rand((1, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False).cuda()
        self.GATENet = GATENet(self.chan_num * self.chan_num, reduction_ratio=128)

        self.fc1 = nn.Linear(self.chan_num*self.band_num*2,64)
        self.fc2 = nn.Linear(64,self.class_num)

    def forward(self,x):
        
        A_ds = self.GATENet(self.A)
        A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        de = x[:,:,:self.band_num]
        psd = x[:,:,self.band_num:]
        feat1 = self.gcn_DE(de,A_ds),A_ds)
        feat2 = self.gcn_PSD(psd,A_ds),A_ds)
        
        feat = torch.cat([feat1,feat2],dim=2)
        feat = self.ATFFNet(feat,A_ds)
        feat = feat.reshape(-1,self.chan_num*self.band_num*2)
        feat = self.linear(feat)
        out = self.linear2(feat)
        
        
        return out
    
    

if __name__ =='main':
    for dataset in ['seed','seed iv']:
        model = DANet(dataset).cuda()
        x = torch.rand(10,62,10).cuda()
        output = model(x)
        print('Output for {} Dataset: '.format(dataset), output.shape)
