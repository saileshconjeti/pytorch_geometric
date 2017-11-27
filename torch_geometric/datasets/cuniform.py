import os

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.ply import read_ply
from ..graph.geometry import edges_from_faces


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

        self.process()

        file = self.training_file if train else self.test_file
        index, position, target, slice = torch.load(file)
        print(index.size(), index.type())
        print(position.size(), position.type())
        print(target.size(), target.type())
        print(slice.size(), slice.type())

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
        slices = [0]
        positions = []
        targets = []
        for name in self.test_files:
            print(name)
            file_path = os.path.join(self.raw_folder, name)
            position, face, target = read_ply(file_path)
            index = edges_from_faces(face)
            indices.append(index)
            positions.append(position)
            targets.append(target)
            slices.append(target.size(0))

        index = torch.cat(indices, dim=1)
        position = torch.cat(positions, dim=0)
        target = torch.cat(targets, dim=0)
        slice = torch.LongTensor(slices)

        file_path = os.path.join(self.processed_folder, 'test.pt')
        torch.save((index, position, target, slice), file_path)
