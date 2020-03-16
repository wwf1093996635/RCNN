import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.optim as optim
import numpy as np

from RCNN_LRN_S import RCNN
#from RCNN_BN_D import RCNN
#from rcnn import RCNN

def prepare_MNIST(device, load=False, model_name=''):
    if(load==False):   
        net=RCNN(in_channels=1,feature_num=32,feature_map_width=28,iter_time=3,device=device)
    else:
        net=torch.load(model_name)
        print('loading model:'+model_name)
    
    batch_size=64
    batch_size_test=64

    net.train()    

    transform = transforms.Compose(
    [transforms.ToTensor()]) 

    #trainset = torchvision.datasets.MNIST(root='D:\\Software_projects\\RCNN\\data\\MNIST', transform=transform, train=True, download=False)
    #testset = torchvision.datasets.MNIST(root='D:\\Software_projects\\RCNN\\data\\MNIST', transform=transform, train=False, download=False)

    trainset = torchvision.datasets.MNIST(root='./data', transform=transform, train=True, download=False)
    testset = torchvision.datasets.MNIST(root='./data', transform=transform, train=False, download=False)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    testloader = DataLoader(dataset=testset, batch_size=batch_size_test, shuffle=True, num_workers=16)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1.00e-04, weight_decay=1.00e-04)
    net=net.to(device)
    return batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer

def prepare_CIFAR_10(device, load=False, model_name='', augment=True):
    if(load==False):   
        net=RCNN(in_channels=3,feature_num=96,feature_map_width=32,iter_time=3,device=device)
        #net = RCNN(3, 10, 96)
    else:
        net=torch.load(model_name)
        print('loading model:'+model_name)

    batch_size=64
    batch_size_test=64

    if(augment==True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.TenCrop(24),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops]))
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    #trainset = torchvision.datasets.CIFAR10(root='D:\\Software_projects\\RCNN\\data\\CIFAR_10', train=True, transform=transform, download=False)
    #testset = torchvision.datasets.CIFAR10(root='D:\\Software_projects\\RCNN\\data\\CIFAR_10',train=False, transform=transform, download=False)

    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR_10', train=True, transform=transform_train, download=False)
    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR_10',train=False, transform=transform_test, download=False)
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    testloader = DataLoader(dataset=testset, batch_size=batch_size_test, shuffle=True, num_workers=16)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.00e-02, momentum=0.9, weight_decay=1.00e-05, nesterov=True)
    #optimizer = optim.Adam(net.parameters(), lr=1.00e-03, weight_decay=1.00e-05)
    net=net.to(device)
    return batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer

def evaluate(net, testloader, criterion, batch_size, report_size, epoch, scheduler, device, augment):
    count=0
    labels_count=0
    correct_count=0
    labels_count=0
    current_labels_count=0
    correct_correct_count=0
    val_loss=0.0
    val_acc=0.0
    net.eval()
    print('validating', end=' ')
    for data in testloader:
        torch.cuda.empty_cache()
        inputs, labels = data
        count=count+1
        inputs=inputs.to(device)
        labels=labels.to(device)
        if(augment==True):
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs = net(inputs)
            outputs = outputs.to(device)  
        
        val_loss += criterion(outputs, labels).item()
        current_correct_count=(torch.max(outputs.data, 1)[1]==labels).sum().item()
        current_labels_count=labels.size(0)
        
        labels_count+=current_labels_count
        correct_count+=current_correct_count
        
        val_acc+=current_correct_count/(current_labels_count*1.0)
    val_acc/=(count)
    print('val_loss:%.4f val_acc:%.4f'%(val_loss/count,correct_count/(labels_count*1.0)))
    scheduler.step(val_loss)
    net.train()

def main():
    print(torch.__version__)
    if(torch.cuda.is_available()==True):
        print("cuda is available")
    else:
        print("cuda is not available")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    augment = True

    #MNIST
    #batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_MNIST(device, load=False,model_name='./RCNN--CIFAR10-model/RCNN-CIFAR10-epoch-26.pth')
    
    #CIFAR-10
    batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_CIFAR_10(device, load=False, augment=augment)
    #batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_CIFAR_10(device, load=True, model_name='./RCNN-CIFAR10-model/RCNN-CIFAR10-epoch-8.pth')
    
    epochs=200
    #scheduler = MultiStepLR(optimizer, milestones=[int(epochs/2),int(epochs*3/4),int(epochs*7/8)], gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1.00e-04, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
    report_size=100
    report_size_test=10

    epoch_0=1

    net.train()

    print('start training')

    if(epoch_0==1):
        with torch.no_grad():
            evaluate(net,testloader,criterion,batch_size_test,report_size_test,-1,scheduler,device,augment)
        torch.save(net, './RCNN-CIFAR10-model/RCNN-CIFAR10-epoch-(-1).pth')   
     
    
    for epoch in range(epoch_0,200):
        torch.cuda.empty_cache()
        print('epoch=%d'%(epoch), end=' ')
        loss_total = 0.0
        acc=0.0
        current_labels_count=0
        current_correct_count=0
        labels_count=0
        correct_count=0
        count=0
        for i, data in enumerate(trainloader, 0):
            count=count+1

            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            # zeros the paramster gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)  
            loss.backward()     
            optimizer.step()

            current_correct_count=(torch.max(outputs,1)[1]==labels).sum().item()
            current_labels_count=labels.size(0)
            
            labels_count+=current_labels_count
            correct_count+=current_correct_count
            
            acc+=current_correct_count/(current_labels_count*1.0)
            loss_total += loss.item()
            
            '''
            if(current_labels_count!=batch_size):
                print("current_labels_count:%d != batch_size:%d"%(current_labels_count, batch_size))
            if i % report_size == (report_size-1):
                print('epoch: %d, correct_count: %d labels_count:%d loss: %.4f acc: %.4f ' %
                    (epoch, correct_count, labels_count, loss_total / report_size, acc / report_size)) 
                loss_total = 0.0
                acc=0.0
            '''
        print('train_loss:%.4f train_acc:%.4f'%(loss_total/(count*1.0), correct_count/(labels_count*1.0)), end=' ')
        torch.save(net, './RCNN-LN-CIFAR10-model/RCNN-CIFAR10-epoch-%d.pth'%(epoch))
        #torch.save(net, 'D:\\Software_projects\\RCNN\\RCNN-saved-epoch-%d.pth'%(epoch))
        #print('Module Saved')

        with torch.no_grad():
            evaluate(net,testloader,criterion,batch_size_test,report_size_test,epoch,scheduler,device,augment)
        
    print('Finished Training')

if __name__ == '__main__':
    main()
    # print(__name__)