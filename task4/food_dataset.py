import numpy as np
from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import transforms
import config
import torch

"""
Initialize food class attributes.
        Args:
            root: The root to the image folder.
            triplets: the triplet sets
"""

class FoodDataset(Dataset):
    def __init__(self, root, triplets, train = True, val = False, test = False):
        self._data = {
            'root': root,
            'triplets':triplets,
            'image_triplets': None,
            'suffix': ".jpg"
        }
        self.load_image_triplets(root, triplets)
        self.train = train
        self.val = val
        self.test = test
        if self.train:
            # if training
            self.transform = config.TRANSFORM_TRAIN
        else:
            # if validation or testing
            self.transform = config.TRANSFORM_TEST

    def load_image_triplets(self, root, triplets):
        image_triplets = {'a': [], 'b': [], 'c': []}
        for t in triplets:
            image_triplets['a'].append(self._data['root'] + '/' + "{:0>5d}".format(int(t[0])) + self._data['suffix'])
            image_triplets['b'].append(self._data['root'] + '/' + "{:0>5d}".format(int(t[1])) + self._data['suffix'])
            image_triplets['c'].append(self._data['root'] + '/' + "{:0>5d}".format(int(t[2])) + self._data['suffix'])
        self._data['image_triplets'] = image_triplets

    def __getitem__(self, idx):
        img_a = self._data['image_triplets']['a'][idx]
        img_b = self._data['image_triplets']['b'][idx]
        img_c = self._data['image_triplets']['c'][idx]
        if self.transform:
            img_a = self.transform(Image.open(img_a))
            img_b = self.transform(Image.open(img_b))
            img_c = self.transform(Image.open(img_c))
        return (img_a, img_b, img_c),()

    def __len__(self):
        assert len(self._data['image_triplets']['a']) == len(self._data['image_triplets']['b'])
        assert len(self._data['image_triplets']['b']) == len(self._data['image_triplets']['c'])
        return len(self._data['image_triplets']['a'])

