from __future__ import division, print_function

import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import Cuniform  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'Cuniform')
train_dataset = Cuniform(path, train=True)
test_dataset = Cuniform(path, train=False)
