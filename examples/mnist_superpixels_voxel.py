from __future__ import division, print_function

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import MNISTSuperpixels  # noqa
from torch_geometric.transforms import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import MaxPoolVoxel  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'MNISTSuperpixels')
transform = CartesianAdj()
train_dataset = MNISTSuperpixels(path, train=True, transform=transform)
test_dataset = MNISTSuperpixels(path, train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.c = torch.cuda.LongTensor([7, 7])

    def forward(self, x, adj, position):
        x = F.elu(self.conv1(adj, x))
        print('a', x.size())
        x, adj, position = MaxPoolVoxel(adj, position, self.c, 49)(x)
        print('b', x.size())
        x = F.elu(self.conv2(adj.data, x))
        x = x.mean(dim=0).view(1, -1)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for data in train_loader:
        input, target, = data['input'].view(-1, 1), data['target']
        adj, position = data['adj']['content'], data['position']

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
            adj, position = adj.cuda(), position.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(input, adj, position)
        loss = F.nll_loss(output, target[:1])
        loss.backward()
        optimizer.step()
        print(loss.data[0])


train(0)
