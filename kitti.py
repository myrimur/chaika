import os
from torchvision.io import read_image, ImageReadMode
import numpy as np


# Normalized left color camera's intrinsics
P_rect_02 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                      [0.000000e+00, 7.215377e+02, 1.728540e+02],
                      [0.000000e+00, 0.000000e+00, 1.000000e+00]]) / np.array([1242, 375, 1])[:, np.newaxis]


class KittiRaw:
    """Dataset wrapper to read raw KITTI data."""
    def __init__(self, path, transform=None):
        self._path = path
        self._transform = transform
        self._len = len(os.listdir(os.path.join(path, 'image_02/data')))

    def __getitem__(self, idx):
        image = read_image(os.path.join(self._path, 'image_02/data', f'{idx:010}.png'), ImageReadMode.RGB)
        if self._transform:
            image = self._transform(image)

        return image

    def __len__(self):
        return self._len
