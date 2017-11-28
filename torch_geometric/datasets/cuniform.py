import os

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.ply import read_ply
from ..graph.geometry import edges_from_faces
from ..sparse import SparseTensor
from .data import Data


class Cuniform(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Cuniform, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.transform = transform

        # Process data.
        self.process()

        # Load processed data.
        file = self.training_file if train else self.test_file
        index, position, target, node_slice, edge_slice = torch.load(file)
        self.index, self.position, self.target = index, position, target.long()
        self.node_slice, self.edge_slice = node_slice, edge_slice

    def __getitem__(self, i):
        start, end = self.edge_slice[i], self.edge_slice[i + 1]
        index = self.index[:, start:end]
        weight = torch.ones(index.size(1))

        start, end = self.node_slice[i], self.node_slice[i + 1]
        position = self.position[start:end]
        target = self.target[start:end]

        n = position.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))
        data = Data(None, adj, position, target)

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return self.node_slice.size(0) - 1

    @property
    def _processed_exists(self):
        return os.path.exists(self.processed_folder)

    def process(self):
        if self._processed_exists:
            return

        self.files = sorted(os.listdir(self.raw_folder))
        self.training_files = self.files[:70]
        self.test_files = self.files[70:]

        make_dirs(os.path.join(self.processed_folder))

        indices = []
        node_slices = [0]
        edge_slices = [0]
        positions = []
        targets = []
        for name in self.training_files:
            print(name)
            file_path = os.path.join(self.raw_folder, name)
            position, face, target = read_ply(file_path)
            index = edges_from_faces(face)
            indices.append(index)
            positions.append(position)
            targets.append(target)
            node_slices.append(node_slices[-1] + target.size(0))
            edge_slices.append(edge_slices[-1] + index.size(1))

        index = torch.cat(indices, dim=1)
        position = torch.cat(positions, dim=0)
        target = torch.cat(targets, dim=0)
        node_slice = torch.LongTensor(node_slices)
        edge_slice = torch.LongTensor(edge_slices)

        file_path = os.path.join(self.processed_folder, 'training.pt')
        data = (index, position, target, node_slice, edge_slice)
        torch.save(data, file_path)
