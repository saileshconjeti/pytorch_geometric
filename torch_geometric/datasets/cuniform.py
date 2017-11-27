import os

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.progress import Progress


class Cuniform(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Cuniform, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.train = train
        self.transform = transform
        print(self.root)

        self.process()

    @property
    def _processed_exists(self):
        return False
        return os.path.exists(self.processed_folder)

    def process(self):
        if self._processed_exists:
            return

        make_dirs(os.path.join(self.processed_folder))
