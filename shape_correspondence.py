import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch_geometric.datasets.faust import FAUST
from torch_geometric.graph.geometry import EuclideanAdj
from torch_geometric.utils.dataloader import DataLoader
from torch_geometric.nn.modules import SplineGCN, Lin

path = '~/MPI-FAUST'
train_dataset = FAUST(
    path, train=True, correspondence=True, transform=EuclideanAdj())
test_dataset = FAUST(
    path, train=False, correspondence=True, transform=EuclideanAdj())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(
            1, 32, dim=3, kernel_size=[3, 3, 3], is_open_spline=True)
        self.conv2 = SplineGCN(
            32, 64, dim=3, kernel_size=[3, 3, 3], is_open_spline=True)
        self.conv3 = SplineGCN(
            64, 128, dim=3, kernel_size=[3, 3, 3], is_open_spline=True)
        self.lin1 = Lin(128, 256)
        self.lin2 = Lin(256, 6890)

    def forward(self, adj, x):
        x = F.relu(self.conv1(adj, x))
        x = F.relu(self.conv2(adj, x))
        x = F.relu(self.conv3(adj, x))
        x = F.relu(self.lin1(x))
        # x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    for batch_idx, ((_, (adj, _)), target) in enumerate(train_loader):
        print('epoch: ', epoch, 'batch: ', batch_idx)
        features = torch.ones(adj.size(0)).view(-1, 1)

        if torch.cuda.is_available():
            features, adj, target = features.cuda(), adj.cuda(), target.cuda()

        features, target = Variable(features), Variable(target)
        output = model(adj, features)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target.view(-1), size_average=True)
        loss.backward()
        optimizer.step()

        # pred = output.data.max(1, keepdim=True)[1]
        # correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        # print('epoch', epoch, 'batch', batch_idx, 'loss', l, 'correct',
        #       correct)
        # optimizer.step()

    # print('Train Epoch: {}\tLoss: {:6f}'.format(epoch, loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0

    for (_, (adj, _)), target in test_loader:
        n = adj.size(0)
        adj = torch.sparse.FloatTensor(adj._indices(),
                                       torch.ones(adj._indices().size(1)),
                                       torch.Size([n, n]))

        x = torch.ones(n).view(-1, 1)
        if torch.cuda.is_available():
            x, adj, target = x.cuda(), adj.cuda(), target.cuda()

        x, adj, target = Variable(x), Variable(adj), Variable(target)
        target = target.view(-1)

        output = model(adj, x)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('Test set: Accuracy: {}/{}'.format(correct, 20 * 6890))


for epoch in range(1, 2):
    train(epoch)
    # test()