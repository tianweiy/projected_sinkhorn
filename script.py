from __future__ import print_function
import sys
sys.path.append('./pytorch-cifar')
sys.path.append('./pytorch-cifar/models')


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model import *
from pgd import attack

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--reg', default=3000, type=float,
                    help='entropy regularization')
parser.add_argument('--p', default=2, type=float, help='p-wasserstein distance')
parser.add_argument('--alpha', default=0.1, type=float, help='PGD step size')
parser.add_argument('--norm', default='linfinity')
parser.add_argument('--ball', default='wasserstein')
parser.add_argument('--checkpoint')
args = parser.parse_args()

if args.checkpoint is None:
    raise ValueError('Need checkpoint file to attack')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

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
mu = torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(-1).unsqueeze(-1).to(device)
std = torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(-1).unsqueeze(-1).to(device)
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model architecture is from pytorch-cifar submodule
print('==> Building model..')
net = CW2_Net()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

regularization = args.reg

path = "/home/yintianwei2000/NFP/state_dict-ep_62.pth"
net.load_state_dict(torch.load(path))

# checkpoint_name = './checkpoints/{}'.format(args.checkpoint)
#save_name = './epsilons/{}_reg_{}_p_{}_alpha_{}_norm_{}_ball_{}.pth'.format(
#                args.checkpoint, regularization, args.p,
#                args.alpha, args.norm, args.ball)
# Load checkpoint.
print('==> Resuming from checkpoint..')
#checkpoint = torch.load(checkpoint_name)
#net.load_state_dict(checkpoint['net'])

# freeze parameters
for p in net.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()

print('==> regularization set to {}'.format(regularization))
print('==> p set to {}'.format(args.p))


def test_attack():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_epsilons = []

    adv_images = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_pgd, _, epsilons = attack(torch.clamp(unnormalize(inputs),min=0),
                                         targets, net,
                                         normalize=normalize,
                                         regularization=regularization,
                                         p=args.p,
                                         alpha=args.alpha,
                                         norm = args.norm,
                                         ball = args.ball,
                                         epsilon = 0.001,
                                         epsilon_factor=1.17,
                                         maxiters=400)

        outputs_pgd = net(normalize(inputs_pgd))
        loss = criterion(outputs_pgd, targets)

        test_loss += loss.item()
        _, predicted = outputs_pgd.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        mask = (targets == predicted)
        adv_images.append(torch.masked_select(inputs_pgd, mask))

    save_path = "./images"
    torch.save(adv_images, save_path)

print('==> Attacking model..')
test_attack()
