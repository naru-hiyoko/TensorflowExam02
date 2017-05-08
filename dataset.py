
import os
import sys

import skimage
from skimage.io import imread
from  skimage.transform import resize

import numpy as np

prefix = './data/img'

class Data:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.count = -1
        self.epoch = 0
        pass


    def next_batch(self, n):
        self.count += 1
        _from = self.count * n
        _to = (self.count + 1) * n
        xs, ys = (self.images[_from: _to], self.labels[_from: _to])

        if self.images.shape[0] <= self.count * n:
            self.count = -1
            self.epoch += 1
            return self.next_batch(n)
        else:
            return (xs, ys)


class SingleFontData:
    def __init__(self):

        data = []
        ids = []
        files = os.listdir(prefix)
        for file in files:
            if '.jpg' in file:
                id = int(file.split('_')[0])
                img = imread(os.path.join(prefix, file), as_grey=True)
                img = resize(img, [64, 64])
                img = np.reshape(img, [64, 64, 1])
                data.append(img)

                _id = np.zeros([10])
                _id[id] = 1.0
                ids.append(_id)
        data = np.asarray(data, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int32)

        r = [x for x in range(ids.shape[0])]
        np.random.shuffle(r)

        ratio = int(ids.shape[0] * 0.8)
        self.train = Data(data[r[:ratio]], labels=ids[r[:ratio]])
        self.test = Data(data[r[ratio:]], labels=ids[r[ratio:]])
