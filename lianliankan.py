# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np

from cv2 import cv2
from getopt import getopt
from itertools import combinations

from src.game import BaseGame


class LianLianKan(BaseGame):
    SOURCE_DIR = 'lianliankan'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.items = []
        self.color_0_0 = None
        self.screencap_interval = 3

        for i in range(1, 14):
            img = self.imread('items/item_%d.jpg' % i)
            self.items.append(img)

        self.imgs = {
            'goon': self.imread('goon.png'),
            'next': self.imread('next.png'),
            'cancel': self.imread('cancel.png'),
            'accept': self.imread('accept.png'),
            'redpack': self.imread('redpack.png'),
            'close_0': self.imread('close_0.png'),
            'close_1': self.imread('close_1.png'),
            'bin_close_0': self.imread('bin_close_0.png', 0)
        }

        self.btns = [
            self.imgs['next'],
            (self.imgs['close_0'], slice(-200), None, None, 0.9),
            (self.imgs['close_1'], slice(200)),
            self.imgs['accept'],
            (self.imgs['cancel'], slice(-200, None)),
            (self.imgs['redpack'], None, None, (0, 440)),
            (self.imgs['bin_close_0'], slice(200), True)
        ]

        self.inters = [
            self.imgs['goon']
        ]

    def run(self):
        pos = None

        while True:
            bmp = self.screencap()
            pos, items = self.initialization(bmp, pos)

            print()
            print(pos)

            while True:
                for pts in items:
                    for pt1, pt2 in combinations(pts, 2):
                        index1 = self.get_pos(pos, pt1)
                        index2 = self.get_pos(pos, pt2)
                        
                        if not self.check_link(pos, index1, index2):
                            continue

                        self.adb.tap(pt1)
                        self.adb.tap(pt2)
                        pts.remove(pt1)
                        pts.remove(pt2)
                        pos[index1[1], index1[0]] = 0
                        pos[index2[1], index2[0]] = 0
                        break
                    else:
                        continue
                    break
                else:
                    break
        
            time.sleep(5)
                
    def initialization(self, bmp, pos=None):
        items = []

        if pos is None:
            pos = np.zeros((3, 3), int)
            pos[2, 0], pos[0, 2] = bmp.shape[:2]
        else:
            pos[1:, 1:] = 0

        for _id, item in enumerate(self.items):
            pts = []

            for match in self.find_all_template(bmp, item):
                pts.append(match['pt'])
                pos = self.set_pos(pos, pts[-1], _id + 1)
            
            if pts:
                items.append(pts)

        return pos, items

    # 点击按钮
    def tap_btn(self, btn, bmp, offset=None, threshold=0.8):
        match = self.find_template(bmp, btn, threshold)

        if match is not None:
            pt = match['pt']

            if offset is not None:
                x, y = offset
                pt = pt[0] + x, pt[1] + y

            self.tap(pt)
            time.sleep(3)
            return True

        return False

    # 截屏
    def screencap(self, check=False):
        bmp = None

        while True:
            bmp = super().screencap()

            # 参与互动
            for inter in self.inters:
                match = self.find_template(bmp, inter)
                
                if match is not None:
                    x1, y1 = match['rect'][0]
                    x2, y2 = match['rect'][1]
                    temp = bmp[y1:y2, x1:x2]
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                    _, temp = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY)

                    if (temp == 0).all():
                        self.tap(match['pt'])
                        time.sleep(3)
                        break
            else:
                # 点击按钮
                for btn in self.btns:
                    temp = bmp
                    offset = [0, 0]
                    is_gray = False
                    match_threshold = 0.8

                    if isinstance(btn, tuple):
                        if len(btn) >= 2 and btn[1] is not None:
                            temp = temp[btn[1]]

                            if btn[1].start is not None:
                                offset[1] += btn[1].start % bmp.shape[0]

                        if len(btn) >= 3 and btn[2] is not None:
                            is_gray = btn[2]
                        if len(btn) >= 4 and btn[3] is not None:
                            offset[0] += btn[3][0]
                            offset[1] += btn[3][1]
                        if len(btn) >= 5 and btn[4] is not None:
                            match_threshold = btn[4]
                        btn = btn[0]
                    
                    # self.imwrite('test.png', temp)
                    # self.showImg(temp)
                    
                    if is_gray:
                        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

                        for threshold in [160, 180, 200]:
                            _, temp_bin = cv2.threshold(temp, threshold, 255, cv2.THRESH_BINARY)

                            # self.imwrite('test.png', temp_bin)
                            # self.showImg(temp_bin)

                            if self.tap_btn(btn, temp_bin, offset, match_threshold):
                                break
                        else:
                            continue
                    elif not self.tap_btn(btn, temp, offset, match_threshold):
                        continue
                    break
                else:
                    if self.color_0_0 is None:
                        self.color_0_0 = bmp[0, 0]
                        break
                    elif (bmp[0, 0] == self.color_0_0).all():
                        break

        return bmp

    # 读取图片
    def imread(self, path, flags=1, join=True):
        if join:
            path = os.path.join(self.SOURCE_DIR, path)

        return super().imread(path, flags)

    # 检查是否可链接
    @classmethod
    def check_link(cls, pos, index1, index2):
        if cls.check_link_0(pos, index1, index2):
            return True

        if cls.check_link_1(pos, index1, index2):
            return True

        if cls.check_link_2(pos, index1, index2):
            return True

        return False

    # 检查是否无拐点链接
    @classmethod
    def check_link_0(cls, pos, index1, index2):
        i, j = index1
        p, q = index2

        if i == p:
            pos = pos[min(j, q) + 1:max(j, q), i]
        elif j == q:
            pos = pos[j, min(i, p) + 1:max(i, p)]
        else:
            return False

        for val in pos:
            if val != 0:
                return False

        return True

    @classmethod
    def check_link_1(cls, pos, index1, index2):
        i, j = index1
        p, q = index2

        if pos[q, i] == 0:
            check1 = cls.check_link_0(pos, index1, (i, q))
            check2 = cls.check_link_0(pos, index2, (i, q))
            
            if check1 and check2:
                return True
        elif pos[j, p] == 0:
            check1 = cls.check_link_0(pos, index1, (p, j))
            check2 = cls.check_link_0(pos, index2, (p, j))
            
            if check1 and check2:
                return True
        
        return False

    @classmethod
    def check_link_2(cls, pos, index1, index2):
        # 上
        i, j = index1

        while j > 1:
            j -= 1

            if pos[j, i] != 0:
                break

            if cls.check_link_1(pos, (i, j), index2):
                return True

        # 右
        i, j = index1

        while i < pos.shape[1] - 1:
            i += 1

            if pos[j, i] != 0:
                break

            if cls.check_link_1(pos, (i, j), index2):
                return True

        # 下
        i, j = index1

        while j < pos.shape[0] - 1:
            j += 1

            if pos[j, i] != 0:
                break

            if cls.check_link_1(pos, (i, j), index2):
                return True

        # 右
        i, j = index1

        while i > 1:
            i -= 1

            if pos[j, i] != 0:
                break

            if cls.check_link_1(pos, (i, j), index2):
                return True

        return False

    # 动态设置pos值
    @staticmethod
    def set_pos(pos, pt, value, r=40):
        i = j = 0

        for i, x in enumerate(pos[0]):
            if pt[0] > x + r:
                continue

            if pt[0] < x - r:
                pos = np.insert(pos, i, 0, 1)
                pos[0, i] = pt[0]

            break

        for j, y in enumerate(pos):
            if pt[1] > y[0] + r:
                continue

            if pt[1] < y[0] - r:
                pos = np.insert(pos, j, 0, 0)
                pos[j, 0] = pt[1]

            break

        pos[j][i] = value
        return pos

    # 获取index
    @staticmethod
    def get_pos(pos, pt, r=40):
        i = j = 0

        for i in range(2, len(pos[0])):
            if pos[0, i] - r <= pt[0] and pt[0] <= pos[0, i] + r:
                break

        for j in range(2, len(pos)):
            if pos[j, 0] - r <= pt[1] and pt[1] <= pos[j, 0] + r:
                break
        
        return i, j


if __name__ == '__main__':
    device = '127.0.0.1:62029'
    opts, _ = getopt(sys.argv[1:], '-d:', ['device='])

    for name, value in opts:
        if name in ('-d', '--device'):
            device = value

    game = LianLianKan(device)
    game.run()
