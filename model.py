'''VGG11/13/16/19 in Pytorch.'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models
from torch.nn import functional as F
import torch.optim as optim
import os
import numpy as np
import copy
import time
import torchvision.transforms as transforms


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def train(model, criterion, optimizer, trainloader, testloader, epochs=2, log_interval=50, lr=0.1):
    print('----- Train Start -----')
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(trainloader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            output = model(batch_x)

            optimizer.zero_grad()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_history.append(loss.item())
            if step % log_interval == (log_interval-1):
                print('[%d, %5d] loss: %.4f' %
                        (epoch + 1, step + 1, running_loss / log_interval))
                running_loss = 0.0
        model.eval()
        print('------ Test Start -----')
        train_acc_history.append(test(model, trainloader))
        test_acc_history.append(test(model, testloader))
        if test_acc_history[-1]>best_acc:
            best_acc = test_acc_history[-1]
            best_epoch = epoch+1
            print('Saving..')
            state = model.state_dict(),
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/r_%s_ckpt.pth' % (lr))


    print('----- Train Finished -----')
    return {
        'loss_history':loss_history,
        'train_acc_history':train_acc_history,
        'test_acc_history':test_acc_history,
        "best_epoch":best_epoch,
        "best_acc":best_acc,
    }

def test(model, testloader):
    # print('------ Test Start -----')

    correct = 0
    total = 0

    with torch.no_grad():
        for test_x, test_y in testloader:
            images, labels = test_x.cuda(), test_y.cuda()
            output = model(images)
            # print(output)
            _, predicted = torch.max(output.data, 1)
            # _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # print('Accuracy of the network is: %.4f %%' % accuracy)

    return accuracy

def PSO_train(models, criterion, epochs, Wmax, Wmin, c1, c2, trainloader, testloader):
    '''
    :param models:
    :param epochs:
    :param Wmax:
    :param Wmin:
    :param c1:
    :param c2:
    :param trainloader:
    :param testloader:
    :return:
    '''
    num_models = len(models)
    max_V = torch.tensor(0.1).cuda()
    current_V = {} #当前网络参数的速度
    current_postion = {}  #当前网络参数的映射
    best_postion = {} #每个网络最好状态参数的存储
    current_acc = []  #当前迭代论述精度值
    best_acc = []     #当前每个网络精度值
    best_loss = []    #用loss代替精度
    best_globel = 0   #全局最优网络所在位置
    Wmax = Wmax
    Wmin = Wmin
    c1 = c1
    c2 = c2
    c = list(models[0].state_dict().keys()) #网络的参数字典

    train_acc_history = []
    test_acc_history = []
    '''
    网络参数初始化
    '''
    for i in range(num_models):
        models[i].apply(weights_init)
        # state = models[i].state_dict(),  #将当前的参数视为最好状态保存
        # if not os.path.isdir('checkpoint_pso'):
        #     os.mkdir('checkpoint_pso')
        # torch.save(state, './checkpoint_pso/%s_ckpt.pth' % (i))
        for key in c:
            if key.endswith('weight'):
                current_postion['%s'%i + key] = models[i].state_dict()[key]
                best_postion['%s' % i + key] = copy.deepcopy(models[i].state_dict()[key])
                current_V['%s'%i + key] = torch.randn(models[i].state_dict()[key].shape).cuda()
            elif key.endswith('bias'):
                current_postion['%s' % i + key] = models[i].state_dict()[key]
                best_postion['%s' % i + key] = copy.deepcopy(models[i].state_dict()[key])
                current_V['%s' % i + key] = torch.randn(models[i].state_dict()[key].shape).cuda()
        models[i].eval()
        current_acc.append(test(models[i], testloader))
        best_acc.append(test(models[i], testloader))
        best_globel = np.argmax(current_acc)
        best_loss.append(float('inf'))
    print('initial acc:', current_acc)

    '''
    开始更新
    '''
    time_start = time.time()
    for epoch in range(epochs):
        w = Wmax - epoch*(Wmax-Wmin)/epochs
        time_step_start = time.time()
        for step, (batch_x, batch_y) in enumerate(trainloader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            for i in range(num_models):
                for key in c:
                    if key.endswith('weight'):
                        current_V['%s' % i + key] = w * current_V['%s' % i + key] + c1 * torch.rand(1).cuda()*(
                                    best_postion['%s' % i + key] - current_postion[
                                '%s' % i + key]) + c2 * torch.rand(1).cuda()*(
                                                                best_postion['%s' % best_globel + key] -
                                                                current_postion['%s' % i + key])
                        current_V['%s' % i + key][current_V['%s' % i + key]>max_V] = max_V
                        current_V['%s' % i + key][current_V['%s' % i + key] < (-max_V)] = -max_V
                        b = current_postion['%s' % i + key] + current_V['%s' % i + key]
                        current_postion['%s' % i + key].copy_(b)
                    elif key.endswith('bias'):
                        current_V['%s' % i + key] = w * current_V['%s' % i + key] + c1 * torch.rand(1).cuda()*(
                                best_postion['%s' % i + key] - current_postion[
                            '%s' % i + key]) + c2 * torch.rand(1).cuda()*(
                                                            best_postion['%s' % best_globel + key] -
                                                            current_postion['%s' % i + key])
                        current_V['%s' % i + key][current_V['%s' % i + key] > max_V] = max_V
                        current_V['%s' % i + key][current_V['%s' % i + key] < (-max_V)] = -max_V
                        # print(current_V['%s' % i + key])
                        b = current_postion['%s' % i + key] + current_V['%s' % i + key]
                        current_postion['%s' % i + key].copy_(b)
                '''
                计算batch的正确率/loss
                '''
                models[i].eval()
                output = models[i](batch_x)
                _, predicted = torch.max(output.data, 1)
                total = batch_y.size(0)
                correct = (predicted == batch_y).sum().item()
                acc = 100 * correct / total
                current_acc[i] = acc
                loss = criterion(output, batch_y).data.cpu().numpy()
                if acc >= best_acc[i]:  #以acc为目标函数
                    best_acc[i] = acc
                # if loss <= best_loss[i]:  #以loss为目标函数
                    best_loss[i] = loss
                    state = models[i].state_dict(),  # 将当前的参数视为最好状态保存
                    if not os.path.isdir('checkpoint_pso'):
                         os.mkdir('checkpoint_pso')
                    torch.save(state, './checkpoint_pso/%s_ckpt.pth' % (i))
                    for key in c:
                        if key.endswith('weight'):
                            best_postion['%s' % i + key] = copy.deepcopy(models[i].state_dict()[key])
                        elif key.endswith('bias'):
                            best_postion['%s' % i + key] = copy.deepcopy(models[i].state_dict()[key])

            # best_globel = np.argmin(best_loss)
            best_globel = np.argmax(best_acc)
            if step % 50 == (50-1):
                time_step_end = time.time()
                print('[%d, %5d] acc:' %
                      (epoch + 1, step + 1), best_acc[best_globel], 'time: %.2f' % (time_step_end - time_step_start))
                print('[%d, %5d] loss:' %
                      (epoch + 1, step + 1), best_loss[best_globel], 'time: %.2f' %(time_step_end-time_step_start))
                time_step_start = time_step_end

        '''
        每个epoch，输出正确率
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # net = VGG("VGG3")
        net = LeNet()
        net.load_state_dict(torch.load('./checkpoint_pso/%s_ckpt.pth' % best_globel)[0])
        net = net.to(device)
        net.eval()
        train_acc = test(net, trainloader)
        train_acc_history.append(train_acc)
        test_acc = test(net, testloader)
        test_acc_history.append(test_acc)
        print('epoch(%d)[train_acc, test_acc]:'%epoch, train_acc, test_acc)

    time_end = time.time()
    print('best_globel:', best_globel)
    print('best loss:', best_loss)
    print('best acc:', best_acc)
    print('total time: %.2f' % (time_end-time_start))
    return {
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
    }
    # state = state = models[best_globel].state_dict(),  # 将当前的参数视为最好状态保存
    # torch.save(state, './checkpoint_pso/best_ckpt.pth')

def PSO_train_HP(models_numbers, interation, Wmax, Wmin, c1, c2, lr_max):

    models_num = models_numbers
    current_V = {}  # 当前参数的速度
    current_postion = {}  # 当前参数的位置
    best_postion = {}  # 每个最好状态参数的存储
    best_globel = 0  # 全局最优网络所在位置
    Wmax = Wmax
    Wmin = Wmin
    c1 = c1
    c2 = c2
    criterion = nn.CrossEntropyLoss()

    test_acc_history = []
    '''
    参数初始化
    '''
    current_V['lr'] = (-1+2*np.random.rand(1, models_num))*lr_max
    current_V['bs'] = ((-1+2*np.random.rand(1, models_num)) * 150).astype(int)
    current_V['epoch'] = ((-1+2*np.random.rand(1, models_num)) * 15).astype(int)
    current_postion['lr'] = lr_max*np.random.rand(1, models_num)
    current_postion['bs'] = (np.random.rand(1, models_num)*300).astype(int)
    current_postion['epoch'] = (np.random.rand(1, models_num)*30).astype(int)
    current_postion['lr'][current_postion['lr'] > lr_max] = lr_max
    current_postion['lr'][current_postion['lr'] <= 0] = 0
    current_postion['bs'][current_postion['bs'] > 300] = 300
    current_postion['bs'][current_postion['bs'] <= 10] = 10
    current_postion['epoch'][current_postion['epoch'] > 30] = 30
    current_postion['epoch'][current_postion['epoch'] <= 0] = 1
    print(current_V)
    print(current_postion)
    best_postion['lr'] = current_postion['lr']
    best_postion['bs'] = current_postion['bs']
    best_postion['epoch'] = current_postion['epoch']
    best_acc = np.zeros([1, models_num])

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
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    start_time = time.time()
    for j in range(interation):
        w = Wmax - (Wmax-Wmin)/interation*j
        for i in range(models_num):
            net = VGG('VGG11')
            net.apply(weights_init)
            net = net.to('cuda')
            lr = current_postion['lr'][0, i]
            epochs = current_postion['epoch'][0, i]
            batch_size = int(current_postion['bs'][0, i])
            print('lr:%s,epochs:%s,batch_size:%s'%(lr, epochs, batch_size))
            # print(type(batch_size))


            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            '''
            每个model的训练
            '''
            for epoch in range(epochs):
                epoch_start_time = time.time()
                net.train()
                for step, (batch_x, batch_y) in enumerate(trainloader):
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                    output = net(batch_x)

                    optimizer.zero_grad()
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

                epoch_end_time = time.time()
                net.eval()
                test_acc = test(net, testloader)
                print('model%s_epoch%s_acc:'%(i+1, epoch+1), test_acc, 'time:%.2f'%(epoch_end_time-epoch_start_time))
            if test_acc >= best_acc[0, i]:
                best_acc[0, i] = test_acc
                best_postion['lr'][0, i] = lr
                best_postion['bs'][0, i] = batch_size
                best_postion['epoch'][0, i] = epochs

                state = net.state_dict(),  # 将当前的参数视为最好状态保存
                if not os.path.isdir('checkpoint_pso_hp'):
                    os.mkdir('checkpoint_pso_hp')
                torch.save(state, './checkpoint_pso_hp/%s_ckpt.pth' % (i))

        best_globel = np.argmax(best_acc)
        test_acc_history.append(np.max(best_acc))
        print('interation_%s_acc:'%(j), best_acc)
        '''
        更新参数
        '''
        if(j < (interation-1)):
            w1 = c1*np.random.rand(1)
            w2 = c1*np.random.rand(1)
            current_V['lr'] = w * current_V['lr'] + w1 * (
                        best_postion['lr'] - current_postion['lr']) - w2 * (
                                          current_postion['lr'] - best_postion['lr'][0, best_globel])
            current_V['lr'][current_V['lr'] > (lr_max/2)] = (lr_max/2)
            current_V['lr'][current_V['lr'] < -(lr_max/2)] = -(lr_max/2)

            current_V['bs'] = w * current_V['bs'] + w1 * (
                    best_postion['bs'] - current_postion['bs']) - w2 * (
                                      current_postion['bs'] - best_postion['bs'][0, best_globel])
            current_V['bs'] = current_V['bs'].astype(int)
            current_V['bs'][current_V['bs'] > 150] = 150
            current_V['bs'][current_V['bs'] <= -150] = -150

            current_V['epoch'] = w * current_V['epoch'] + w1 * (
                    best_postion['epoch'] - current_postion['epoch']) - w2 * (
                                      current_postion['epoch'] - best_postion['epoch'][0, best_globel])
            current_V['epoch'] = current_V['epoch'].astype(int)
            current_V['epoch'][current_V['epoch'] > 15] = 15
            current_V['epoch'][current_V['epoch'] <= -15] = -15

            current_postion['lr'] += current_V['lr']
            current_postion['lr'][current_postion['lr'] > lr_max] = lr_max
            current_postion['lr'][current_postion['lr'] <= 0] = 0
            current_postion['bs'] += current_V['bs']
            current_postion['bs'][current_postion['bs'] > 600] = 600
            current_postion['bs'][current_postion['bs'] <= 10] = 10
            current_postion['epoch'] += current_V['epoch']
            current_postion['epoch'][current_postion['epoch'] > 30] = 30
            current_postion['epoch'][current_postion['epoch'] <= 0] = 1
            print(current_V)
            print(current_postion)
    end_time = time.time()
    print('total time:%.2f'%(end_time-start_time))
    return {
        'test_acc_history': test_acc_history,
        "best_lr": best_postion['lr'][0, best_globel],
        "best_bs": best_postion['bs'][0, best_globel],
        "best_epoch": best_postion['epoch'][0, best_globel],
        "best_acc": np.max(test_acc_history),
    }

if __name__ == "__main__":
    # net = VGG('VGG11')
    net = LeNet()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    # with SummaryWriter(comment='VGG-11') as w:
    #     w.add_graph(net, (x,))
    # print(net.state_dict()['features.0.weight'].shape, net.state_dict()['features.0.bias'].shape)
    # c = list(net.state_dict().keys())
    # a = {}
    # d = {}
    # a['features.0.weight'] = net.state_dict()['features.0.weight']
    # d['features.0.weight'] = net.state_dict()['features.0.weight']
    # print(d['features.0.weight'])
    # b = torch.zeros(net.state_dict()['features.0.weight'].shape)
    # a['features.0.weight'].copy_(b)
    # print(d['features.0.weight'])
    print(net.state_dict())

