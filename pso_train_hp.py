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

result = PSO_train_HP(10, 10, 0.6, 0.1, 2, 2, 0.1)

'''
acc-epoch
'''
plt.figure()
plt.plot(result['test_acc_history'])
plt.xlabel('interation')
plt.ylabel('acc')
plt.title('acc history')
plt.savefig('./checkpoint_pso_hp/acc-interation.png')
plt.show()

print('lr, bs, epoch, acc:', result['best_lr'], result['best_bs'], result['best_epoch'], result['best_acc'])