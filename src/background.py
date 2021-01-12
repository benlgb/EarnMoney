# -*- coding: utf-8 -*-

import numpy as np
from cv2 import cv2
from itertools import product

class BaseBackgroundException(Exception):
    pass


class DifferentShape(BaseBackgroundException):
    def __init__(self, shape):
        msg = 'Could not analysis img with different shape' + str(shape)
        super().__init__(msg)


class TypeError(BaseBackgroundException):
    pass


class NoImageError(BaseBackgroundException):
    pass


class BackgroundAnalysis:
    def __init__(self, history=100, focus=None):
        self.history = history
        self.focus = None
        self.shape = None
        self.imgs = None
        self.color_count = None
        self.background = None

    def __getitem__(self, key):
        if isinstance(key[0], slice) and isinstance(key[1], slice):
            if self.shape is None:
                raise NoImageError('could not slice empty')

            ba = self.__class__()
            ba.background = self.background[key[0],key[1]]
            ba.shape = ba.background.shape
            ba.imgs = self.imgs[:,key[0],key[1]]
            ba.color_count = [line[key[1]] for line in self.color_count[key[0]]]
            return ba
        else:
            raise TypeError('the type of two args must be slice')

    def apply(self, img):
        # 必须要相同的shape
        if self.shape is None:
            self.shape = img.shape
            self.color_count = np.zeros(self.shape[:2]).tolist()

        elif img.shape != self.shape:
            raise DifferentShape(img.shape)

        # 新增图片
        if self.imgs is None:
            self.imgs = np.array([img])
        else:
            self.imgs = np.concatenate((self.imgs, [img]), axis=0)

        # 移除过量图片
        if len(self.imgs) == 100:
            self.imgs = np.delete(self.imgs, 0, axis=0)

        return self._analysis(img)

    def _analysis(self, img):
        self.background = np.zeros(self.shape, np.uint8)

        for y, x in product(range(self.shape[0]), range(self.shape[1])):
            bgr = tuple(img[y,x])
            default = dict(max=0, color=None, focus=None)
            count = self.color_count[y][x] or default
            count[bgr] = count.get(bgr, 0) + 1

            if count[bgr] > count['max']:
                count['max'] = count[bgr]
                count['color'] = img[y,x]

            if self.focus is not None and bgr == self.focus:
                count['focus'] = bgr

            self.background[y,x] = count['focus'] or count['color']
            self.color_count[y][x] = count

        return self.background