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

class RCNN(torch.nn.Module):
    def __init__(self, in_channels, feature_num, iter_time, feature_map_width, device):
        super(RCNN,self).__init__()
        self.feature_num=feature_num

        self.iter_time=iter_time
        self.device=device
      
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=feature_num, kernel_size=3, stride=1, padding=1) 
        #self.conv1 = RCL(in_channels=3,out_channels=feature_num,kernel_size=5,iter_time=0,feature_map_width=32,device=device,stride=1,padding=2)          
                
        self.relu=torch.nn.ReLU()
        self.bn=torch.nn.BatchNorm2d(feature_num)
        #self.ln=torch.nn.LayerNorm([feature_num,32,32])

        #Conv2d will automatically initialize weight and bias.
        #torch.nn.init.kaiming_uniform_(self.conv1[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.kaiming_uniform_(self.conv1[0].bias, a=0, mode='fan_in', nonlinearity='relu')
    
        #print(self.conv1[1].weight)
        #print(self.conv1[1].bias)

        self.mxp1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.dropout1=torch.nn.Dropout(0.5)
        
        self.rconv2 = RCL(in_channels=feature_num,
                          out_channels=feature_num,
                          kernel_size=3,
                          iter_time=3,
                          feature_map_width=feature_map_width//2,
                          device=device,
                          stride=1,
                          padding=1)

        self.dropout2=torch.nn.Dropout(0.5)

        self.rconv3 = RCL(in_channels=feature_num,
                          out_channels=feature_num,
                          kernel_size=3,
                          iter_time=3,
                          feature_map_width=feature_map_width//2,
                          device=device,
                          stride=1,
                          padding=1)

        self.mxp3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.dropout3=torch.nn.Dropout(0.5)

        feature_map_width=feature_map_width//2

        self.rconv4 = RCL(in_channels=self.feature_num,
                          out_channels=self.feature_num,
                          kernel_size=3,
                          iter_time=3,
                          feature_map_width=feature_map_width//2,
                          device=device,
                          stride=1,
                          padding=1)
        
        self.dropout4=torch.nn.Dropout(0.5)

        self.rconv5 = RCL(in_channels=self.feature_num,
                          out_channels=self.feature_num,
                          kernel_size=3,
                          iter_time=3,
                          feature_map_width=feature_map_width//2,
                          device=device,
                          stride=1,
                          padding=1)
        
        self.mxp5 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.dropout5=torch.nn.Dropout(0.5)
        
        self.mlp6 = torch.nn.Linear(self.feature_num,10)
        #self.output = torch.nn.Softmax(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.mxp1(x)
        #x = self.dropout1(x)
        x = self.rconv2(x)
        #x = self.dropout2(x)
        x = self.rconv3(x)
        x = self.mxp3(x)
        x = self.dropout3(x)
        x = self.rconv4(x)
        #x = self.dropout4(x)
        x = self.rconv5(x)
        x = self.mxp5(x)
        x = self.dropout5(x)
        #x = torch.max(torch.max(x,3)[0],2)[0]
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.feature_num)
        
        x=self.mlp6(x)
        return x

class RCL(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, iter_time, feature_map_width, device, stride=1, padding=0):
        super(RCL,self).__init__()
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.iter_time=iter_time

        self.ln=[]
        for i in range(0, iter_time+1):
            self.ln.append(torch.nn.LayerNorm([out_channels,feature_map_width,feature_map_width]))

        for i in range(0, iter_time+1):
            self.ln[i].to(device)
            self.add_module('ln[%d]'%(i),self.ln[i])

        self.relu=torch.nn.ReLU()

        self.conv_f=torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.conv_r=torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, bias=False, stride=self.stride, padding=self.padding)
        
        self.feature_map_width=feature_map_width

        self.feature_map_num=out_channels

    def forward(self, x):
        r=self.conv_r(x)
        r=self.relu(r)
        r=self.ln[self.iter_time](r)
        for i in range(0,self.iter_time):
            r=torch.add(self.conv_r(r),self.conv_f(x))
            r=self.relu(r)
            r=self.ln[i](r)
        return r