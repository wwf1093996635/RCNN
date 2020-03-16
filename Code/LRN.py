
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

class LRN(torch.nn.Module):
    def __init__(self, feature_map_width, out_channels, device):
        super(LRN,self).__init__()
        alpha=torch.zeros(1)
        alpha[0]=0.001
        #self.alpha = torch.nn.Parameter(alpha)
        self.feature_map_num=out_channels
        self.inhiRange=self.feature_map_num//8+1
        inhiMat=torch.zeros(out_channels,out_channels)
        for i in range(0,out_channels):
            j_min= (i-self.inhiRange//2+self.feature_map_num)%self.feature_map_num
            j_max= (i+self.inhiRange//2)%self.feature_map_num
            if(j_min<j_max):
                for j in range(j_min,j_max+1):
                    inhiMat[i,j]=1.0
            else:
                for j in range(0,j_max+1):
                    inhiMat[i,j]=1.0
                for j in range(j_min,self.feature_map_num):
                    inhiMat[i,j]=1.0
        inhi=torch.zeros(640,out_channels,out_channels)
        for i in range(0,640):
            inhi[i]=inhiMat        
        self.inhi=inhi.to(device)
        self.inhiMat=inhiMat.to(device)
    def forward(self,x):
        #print("alpha="+str(self.alpha[0]))
        y=x.clone().detach()
        y=y**2
        y=y.view([x.size(0),x.size(1),x.size(2)*x.size(3)])
        y=torch.bmm(self.inhi[0:y.size(0),:,:],y)
        y=y.view([x.size(0),x.size(1),x.size(2),x.size(3)])
        y=y*0.001/self.inhiRange
        y=y+1.0
        y=y**0.75
        x=x/y
        return x