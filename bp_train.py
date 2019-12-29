import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from model import *

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# learning_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
learning_rate = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
best_epcoh = []
best_acc = []
lr_history_loss=[]
lr_history_best_loss=[]
for lr in learning_rate:
    # Model
    print('==> Building model..')
    net = VGG('VGG11')
    net.apply(weights_init)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    result = train(net, criterion, optimizer, trainloader, testloader, epochs=20, log_interval=50, lr=lr)

    best_epcoh.append(result['best_epoch'])
    best_acc.append(result['best_acc'])
    lr_history_loss.append(result['loss_history'][-1])
    lr_history_best_loss.append(result['loss_history'][best_epcoh[-1]])


    '''
    loss-iteration
    '''
    plt.figure()
    plt.plot(result['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training loss history')
    plt.savefig("fig/lr_%s_learning_curve.png"%(lr))

    '''
    lr-loss-iteration
    '''
    plt.figure(1)
    plt.plot(result['loss_history'], label='ls-%s'%lr)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training loss history')
    plt.legend(loc='upper left')
    plt.savefig("fig/all_learning_curve.png")
    # plt.show()

    '''
    acc-epoch
    '''
    plt.figure()
    plt.plot(result['train_acc_history'], label='train')
    plt.plot(result['test_acc_history'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('acc history')
    plt.legend(loc='upper left')
    plt.savefig("fig/lr_%s_acc.png"%(lr))
    # plt.show()


'''
loss-lr
'''
plt.figure()
plt.plot(learning_rate, lr_history_loss)
plt.xlabel('lr')
plt.ylabel('training loss')
plt.title('loss-lr')
plt.savefig("fig/last_loss_lr.png")

plt.figure()
plt.plot(learning_rate, lr_history_best_loss)
plt.xlabel('lr')
plt.ylabel('training loss')
plt.title('loss-lr')
plt.savefig("fig/best_loss_lr.png")
plt.show()

print("best_epoch:", best_epcoh)
print("best_acc:", best_acc)