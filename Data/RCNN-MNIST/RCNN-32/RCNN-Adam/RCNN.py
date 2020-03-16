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
    def __init__(self, in_channels, feature_num, iter_time, device):
        super(RCNN,self).__init__()
        self.feature_num=feature_num

        self.iter_time=iter_time
        self.device=device
      
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=feature_num, kernel_size=5, stride=1, padding=2)           
        self.bn=torch.nn.BatchNorm2d(feature_num)
        self.relu=torch.nn.ReLU()

        #print(self.conv1[0].weight.shape)
        #print(self.conv1[0].bias.shape)

        #Conv2d will automatically initialize weight and bias.
        #torch.nn.init.kaiming_uniform_(self.conv1[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.kaiming_uniform_(self.conv1[0].bias, a=0, mode='fan_in', nonlinearity='relu')
    
        #print(self.conv1[1].weight)
        #print(self.conv1[1].bias)

        self.mxp1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.rconv2 = RCL(in_channels=feature_num,
                          out_channels=feature_num,
                          kernel_size=3,
                          iter_time=3,
                          stride=1,
                          padding=1)

        self.rconv3 = RCL(in_channels=feature_num,
                          out_channels=feature_num,
                          kernel_size=3,
                          iter_time=3,
                          stride=1,
                          padding=1)

        self.mxp3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.rconv4 = RCL(in_channels=self.feature_num,
                          out_channels=self.feature_num,
                          kernel_size=3,
                          iter_time=3,
                          stride=1,
                          padding=1)

        self.rconv5 = RCL(in_channels=self.feature_num,
                          out_channels=self.feature_num,
                          kernel_size=3,
                          iter_time=3,
                          stride=1,
                          padding=1)
        
        self.mxp5 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.mlp6 = torch.nn.Linear(self.feature_num,10)
        #self.output = torch.nn.Softmax(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mxp1(x)
        x = self.rconv2(x)
        x = self.rconv3(x)
        x = self.mxp3(x)
        x = self.rconv4(x)
        x = self.rconv5(x)
        x = self.mxp5(x)
        y = torch.zeros(x.size(0), x.size(1))
        y = y.to(self.device)
        for i in range(0,x.size(0)):
            for j in range(0,x.size(1)):
                y[i][j]=torch.max(x[i][j])
        y=self.mlp6(y)
        #CrossEntropyLoss already includes softmax.
        #y=F.softmax(y, dim=1)
        return y

class RCL(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, iter_time, stride=1, padding=0):
        super(RCL,self).__init__()
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.forward_weight=torch.nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_uniform_(self.forward_weight, a=0, mode='fan_in', nonlinearity='relu')
    
        self.recurrent_weight=torch.nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_uniform_(self.recurrent_weight, a=0, mode='fan_in', nonlinearity='relu')

        self.forward_bias=torch.nn.Parameter(torch.zeros(out_channels))
 
        self.recurrent_bias=torch.nn.Parameter(torch.zeros(out_channels))
        
        self.bn=torch.nn.BatchNorm2d(out_channels)

        self.relu=torch.nn.ReLU()

        self.conv_f=torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.conv_r=torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.iter_time=iter_time

    def forward(self, x):
        
        f=self.conv_f(x)
        r=f.clone().detach()
        for i in range(0,self.iter_time):
            if(i==0):
                r=self.bn(r)
                r=self.relu(r)
            else:
                r=self.conv_r(r)
                r=torch.add(f,r)
                r=self.bn(r)
                r=self.relu(r)

        return r

        '''
        for i in range(1,self.iter_time+1):
            if (i==1):
                recurrent=feedforward.clone().detach()
                recurrent=self.bn(recurrent)
                outputs=self.relu(recurrent)
            else:
                temp = F.conv2d(outputs, weight=self.recurrent_weight, bias=self.recurrent_bias, stride=self.stride, padding=self.padding)                
                recurrent=torch.add(temp,feedforward)
                
                recurrent=self.bn(recurrent)
                outputs=self.relu(recurrent)
        del feedforward
        return outputs
        '''

def prepare_MNIST(device, load=False, model_name=''):
    if(load==False):   
        net=RCNN(in_channels=1,feature_num=32,iter_time=3,device=device)
    else:
        net=torch.load(model_name)
        print('loading model:'+model_name)
        
    transform = transforms.Compose([transforms.ToTensor()]) 
    batch_size=128
    batch_size_test=128
    net.train()    

    #trainset = torchvision.datasets.MNIST(root='D:\\Software_projects\\RCNN\\data\\MNIST', transform=transform, train=True, download=False)
    #testset = torchvision.datasets.MNIST(root='D:\\Software_projects\\RCNN\\data\\MNIST', transform=transform, train=False, download=False)

    trainset = torchvision.datasets.MNIST(root='./data', transform=transform, train=True, download=True)
    testset = torchvision.datasets.MNIST(root='./data', transform=transform, train=False, download=True)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=batch_size_test, shuffle=True, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1.0000e-06, weight_decay=1.0000e-05)
    net=net.to(device)
    return batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer

def prepare_CIFAR_10(device):
    #CIFAR-10
    batch_size=64
    net=RCNN(in_channels=3,feature_num=64,iter_time=3,device=device)
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), std =(0.5, 0.5, 0.5))]) 

    trainset = torchvision.datasets.CIFAR10(root='D:\\Software_projects\\RCNN\\data\\CIFAR_10', train=True, transform=transform, download=False)
    testset = torchvision.datasets.CIFAR10(root='D:\\Software_projects\\RCNN\\data\\CIFAR_10',train=False, transform=transform, download=False)

    #trainset = torchvision.datasets.CIFAR10(root='~\\wangwf\\RCNN\\data\\CIFAR_10', train=True, transform=transform, download=False)
    #testset = torchvision.datasets.CIFAR10(root='~\\wangwf\\RCNN\\data\\CIFAR_10',train=False, transform=transform, download=False)
    
    
    trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=True, num_workers=0)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net=net.to(device)
    return batch_size, net, trainset, testset, trainloader, testloader, criterion, optimizer

def evaluate(net, testloader, criterion, batch_size, report_size, epoch, scheduler, device):
    count=-1
    labels_count=0
    correct_count=0
    labels_count=0
    current_labels_count=0
    correct_correct_count=0
    val_loss=0.0
    val_acc=0.0
    val_acc_batch=0.0
    net.eval()
    print('validating')
    for data in testloader:
        count=count+1
        torch.cuda.empty_cache()
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = net(inputs)
        outputs = outputs.to(device)
        val_loss += criterion(outputs, labels)
        current_correct_count=(torch.max(outputs,1)[1].cpu().numpy()==labels.cpu().numpy()).sum()
        current_labels_count=labels.cpu().size(0)
        labels_count+=current_labels_count
        correct_count+=current_correct_count
        val_acc_batch+=current_correct_count/(current_labels_count*1.0)
        val_acc+=current_correct_count/(current_labels_count*1.0)
        if(current_labels_count!=batch_size):
            print("current_label_count:%d != batch_size:%d"%(current_labels_count, batch_size))
        if count % report_size == (report_size-1):
            print('epoch: %d, current_count: %d current_correct: %d acc_batch: %.9f' %
                (epoch, labels_count, correct_count, val_acc_batch / report_size))
            val_acc_batch=0.0
    val_loss/=(count+1)
    val_acc/=(count+1)
    print('val_loss:%.9f val_acc:%.9f'%(val_loss,correct_count/(labels_count*1.0)))
    scheduler.step(val_loss)
    net.train()



def main():
    print(torch.__version__)
    if(torch.cuda.is_available()==True):
        print("cuda is available")
    else:
        print("cuda is not available")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #MNIST
    batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_MNIST(device,
        load=True,model_name='./RCNN-model/RCNN-epoch-56.pth')
    
    #CIFAR-10
    #batch_size, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_CIFAR_10(device)
    
    #print(net)
    '''
    params = list(net.parameters())
    print(len(params))
    print(params[2].size())
    '''
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, 
        verbose=True, threshold=1.0000e-04, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    report_size=10
    report_size_test=10

    epoch_0=57

    net.train()
    print('start training')
     
    for epoch in range(epoch_0,200):
        if(epoch>=10):
            report_size=20
        if(epoch>=30):
            report_size=50
        print('epoch=%d'%(epoch))
        running_loss = 0.0
        acc=0.0
        current_labels_count=0
        current_correct_count=0
        labels_count=0
        correct_count=0
        for i, data in enumerate(trainloader, 0):
            torch.cuda.empty_cache()
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            # zeros the paramster gradients
            optimizer.zero_grad()       # 
            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)  
            loss.backward()     
            optimizer.step()
            current_correct_count=(torch.max(outputs,1)[1].cpu().numpy()==labels.cpu().numpy()).sum()
            current_labels_count=labels.cpu().size(0)
            labels_count+=current_labels_count
            correct_count+=current_correct_count
            acc+=current_correct_count/(current_labels_count*1.0)
            
            running_loss += loss.item()
            if(current_labels_count!=batch_size):
                print("current_labels_count:%d != batch_size:%d"%(current_labels_count, batch_size))
            if i % report_size == (report_size-1):
                print('epoch: %d, current_count: %d labels_count:%d loss: %.9f acc: %.9f ' %
                    (epoch, current_labels_count, labels_count, running_loss / report_size, acc / report_size)) 
                running_loss = 0.0
                acc=0.0
                #with torch.no_grad():
                    #evaluate(net,testloader,criterion,batch_size_test,report_size,epoch,scheduler,device)
        
        print('epoch finished. total_labels:%d total_correct:%d total_acc:%.9f'%(labels_count, correct_count, labels_count/(correct_count*1.0)))
        torch.save(net, './RCNN-model/RCNN-epoch-%d.pth'%(epoch))
        #torch.save(net, 'D:\\Software_projects\\RCNN\\RCNN-saved-epoch-%d.pth'%(epoch))
        print('Module Saved')

        with torch.no_grad():
            evaluate(net,testloader,criterion,batch_size_test,report_size_test,epoch,scheduler,device)
        
    print('Finished Training')

    torch.save(net, './RCNN-model/RCNN-epoch-%d.pth'%(epoch))
    #torch.save(net, 'D:\\Software_projects\\RCNN\\RCNN-saved-epoch-%d.pth'%(epoch))

    print('Module Saved')

if __name__ == '__main__':
    main()
    # print(__name__)