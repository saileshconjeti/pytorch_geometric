from __future__ import division, print_function

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cora  # noqa
from torch_geometric.transforms import TargetIndegreeAdj  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa

transform = TargetIndegreeAdj()
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Cora')
dataset = Cora(path, transform=transform)
data = dataset[0]
input, adj, target = data['input'], data['adj'], data['target']
n = adj.size(0)
train_mask = torch.arange(0, n - 1000, out=torch.LongTensor())
test_mask = torch.arange(n - 500, n, out=torch.LongTensor())

if torch.cuda.is_available():
    input, adj, target = input.cuda(), adj.cuda(), target.cuda()
    train_mask, test_mask = train_mask.cuda(), test_mask.cuda()

input, target = Variable(input), Variable(target)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1433, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, 7, dim=1, kernel_size=2)

    def forward(self, adj, x):
        x = F.elu(self.conv1(adj, x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(adj, x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)


def train():
    model.train()

    optimizer.zero_grad()
    output = model(adj, input)
    loss = F.nll_loss(output[train_mask], target[train_mask])
    loss.backward()
    optimizer.step()


def test():
    model.eval()

    output = model(adj, input)
    output = output.data[test_mask]
    pred = output.max(1)[1]
    acc = pred.eq(target.data[test_mask]).sum() / test_mask.size(0)
    return acc


num_runs = 100
acc = torch.zeros(num_runs)

for run in range(1, num_runs + 1):
    model.conv1.reset_parameters()
    model.conv2.reset_parameters()

    for _ in range(0, 200):
        train()

    acc[run - 1] = test()
    print('Run:', run, 'Test Accuracy:', acc[run - 1])

print('Mean:', acc.mean(), 'Stddev:', acc.std())
