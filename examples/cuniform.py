from __future__ import division, print_function

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cuniform  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Cuniform')
transform = CartesianAdj()
train_dataset = Cuniform(path, train=True, transform=transform)
test_dataset = Cuniform(path, train=False, transform=transform)

batch_size = 5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv7 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv8 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv9 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv10 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.lin1 = Lin(64, 128)
        self.lin2 = Lin(128, 4)

    def forward(self, adj, x):
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = F.elu(self.conv4(adj, x))
        x = F.elu(self.conv5(adj, x))
        x = F.elu(self.conv6(adj, x))
        x = F.elu(self.conv7(adj, x))
        x = F.elu(self.conv8(adj, x))
        x = F.elu(self.conv9(adj, x))
        x = F.elu(self.conv10(adj, x))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    model.train()

    # if epoch == 61:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.001

    for data in train_loader:
        adj, target = data['adj']['content'], data['target']
        input = torch.ones(target.size(0), 1)

        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(adj, input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(loss.data[0])


def test(epoch, loader):
    model.eval()

    correct = 0
    total = 0

    for data in loader:
        adj, target = data['adj']['content'], data['target']
        input = torch.ones(target.size(0), 1)

        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        output = model(adj, Variable(input))
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum()
        total += target.size(0)

    print('Epoch:', epoch, 'Accuracy:', correct / total)


for epoch in range(1, 101):
    train(epoch)
    print('Train Validation:')
    test(epoch, train_loader)
    print('Test Validation:')
    test(epoch, test_loader)
