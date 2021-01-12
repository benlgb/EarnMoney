# -*- coding: utf-8 -*-

import re
import time
import uuid
import pytesseract
import numpy as np
from cv2 import cv2
from PIL import Image
from src.adb import ADB

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class BaseGameException(Exception):
    pass


class ClassNameError(BaseException):
    def __init__(self, obj):
        name = obj.__class__.__name__
        super().__init__('Could not use the class name: %s' % name)


class NotFound(BaseException):
    pass


class NoEmptyPosition(BaseException):
    pass


class ImgMatchError(BaseException):
    pass


class BaseGame:
    hsv_color = {
        'white': [
            np.array([0, 0, 255]),
            np.array([180, 0, 255])
        ],
        'black': [
            np.array([0, 0, 0]),
            np.array([180, 0, 0])
        ]
    }

    COLOR_WHITE = 'white'
    COLOR_BLACK = 'black'

    TESSERACT_CONFIG = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/\\:'

    def __init__(self, device):
        self.device = device
        self.adb = ADB(device)
        self.size = self.adb.get_size()
        self.scale = 900 / self.size[0]
        self.name = self.get_class_name()
        self.screencap_time = 0
        self.screencap_img = None
        self.screencap_interval = 1
        self.uid = uuid.uuid3(uuid.NAMESPACE_OID, self.name + device).hex
        self.screencap_name = '%s_%s.png' % (self.name, self.uid)

    # 获取类名
    def get_class_name(self):
        name = self.__class__.__name__
        match = re.match(r'^([A-Z][a-z]*|\d+)+$', name)
        
        if match is None:
            raise ClassNameError(self)
        else:
            parts = re.findall(r'[A-Z][a-z]*|\d+', name)
            parts = [i.lower() for i in parts]
            return '_'.join(parts)

    # 截图
    def screencap(self, interval=None):
        if interval is None:
            interval = self.screencap_interval
            
        while time.time() - self.screencap_time < interval:
            time.sleep(0.1)
            
        self.screencap_time = time.time()
        path = self.adb.screencap(self.screencap_name)
        self.screencap_img = cv2.resize(BaseGame.imread(path), None, fx=self.scale, fy=self.scale)
        return self.screencap_img
    
    # 获取图片文本
    def get_text(self, img, color=None, lowerb=None, upperb=None, reg=None):
        if color is None:
            color = 'black'

        if lowerb is None:
            lowerb = self.hsv_color[color][0]

        if upperb is None:
            upperb = self.hsv_color[color][1]

        temp = Image.fromarray(img)
        text = pytesseract.image_to_string(temp, config=self.TESSERACT_CONFIG)

        if len(img.shape) == 3 and img.shape[2] == 3:
            dilate_kernel = np.ones((2, 2), np.uint8)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bi_img = cv2.inRange(hsv_img, lowerb, upperb)
            bi_img = cv2.dilate(bi_img, dilate_kernel, iterations=1)
            bi_img = Image.fromarray(255 - bi_img)
            text += pytesseract.image_to_string(bi_img, config=self.TESSERACT_CONFIG)

        if reg is None:
            return text
        else:
            match = re.search(reg, text)
            if match is None:
                raise NotFound()
            else:
                return match

    # 检索图片位置（单个）
    @classmethod
    def find_template(cls, source, target, threshold=0.8, ignore=False):
        try:
            result = cls.find_all_template(source, target, threshold, mac_count=1)
        except cv2.error as e:
            if ignore:
                return None
            raise e

        if not result:
            return None

        return result[0]

    # 检索图片位置（多个）
    @classmethod
    def find_all_template(cls, source, target, threshold=0.8, mac_count=None):
        mask = None
        if len(target.shape) == 3 and target.shape[2] == 4:
            mask = target[:, :, 3]
            target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
            
        res = cv2.matchTemplate(source, target, cv2.TM_CCOEFF_NORMED, mask=mask)

        result = []
        height, width = target.shape[:2]
        while True:
            _, max_val, _, tl = cv2.minMaxLoc(res)

            if max_val < threshold:
                break

            br = (tl[0] + width, tl[1] + height)
            mp = (int(tl[0] + width / 2), int(tl[1] + height / 2))
            result.append({
                'pt': mp,
                'rect': (tl, br),
                'conf': max_val
            })

            if mac_count is not None:
                if mac_count <= 0:
                    break
                else:
                    mac_count -= 1

            cv2.floodFill(res, None, tl, (-1000,), max_val-threshold+0.1, 1, flags=cv2.FLOODFILL_FIXED_RANGE)
        return result

    # 二值化
    def threshold(self, img):
        return cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]

    # 自适应阈值处理
    def adaptiveThreshold(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)

    # 显示点
    def showPoint(self, img, point, radius=10, fx=None, fy=None):
        tl = (point[0] - radius, point[1] - radius)
        br = (point[0] + radius, point[1] + radius)
        cv2.rectangle(img, tl, br, (0, 0, 255), 2)
        self.showImg(img, fx=fx, fy=fy)

    # 显示图片
    def showImg(self, img, title='test', wait=True, fx=None, fy=None):
        if fx is not None or fy is not None:
            img = cv2.resize(img, None, fx=fx or fy, fy=fy or fx)
            
        cv2.imshow(title, img)

        if wait:
            cv2.waitKey()

    # 点击
    def tap(self, pt):
        if isinstance(pt, tuple):
            pt = int(pt[0] / self.scale), int(pt[1] / self.scale)
        self.adb.tap(pt)

    # 读取图片
    @classmethod
    def imread(cls, path, flags=1):
        img = np.fromfile(path, np.uint8)
        return cv2.imdecode(img, flags)

    # 保存图片
    @classmethod
    def imwrite(cls, filename, img, params=None, ext='.png'):
        cv2.imencode(ext, img, params)[1].tofile(filename)

    # 打印矩阵
    @classmethod
    def print_matrix(cls, matrix, *args, **kwargs):
        # 参数对外隐藏
        end = length = start = flag_start = None

        try:
            end = args[0]
            length = args[1]
            start = args[2]
            flag_start = args[3]
        except IndexError:
            if end is None:
                end = kwargs.get('end', '\n')

            if length is None:
                length = kwargs.get('length', 0)

            if start is None:
                start = kwargs.get('start', 0)
            
            if flag_start is None:
                flag_start = kwargs.get('flag_start', False)

        if isinstance(matrix, list):
            if flag_start:
                print('[', end='')
            else:
                print(' ' * start + '[', end='')

            if len(matrix) > 0:
                if isinstance(matrix[0], list):
                    # 多维数组
                    max_len = [0 for _ in range(len(matrix[0]))]

                    if len(matrix[0]) > 0 and not isinstance(matrix[0][0], list):
                        for line in matrix:
                            for index, value in enumerate(line):
                                length = cls.get_string_length(value)
                                if length > max_len[index]:
                                    max_len[index] = length

                    if len(matrix) > 1:
                        cls.print_matrix(matrix[0], ',\n', max_len, start + 1, True)
                    else:
                        cls.print_matrix(matrix[0], '', max_len, start + 1, True)

                    for line in matrix[1:-1]:
                        cls.print_matrix(line, ',\n', max_len, start + 1)

                    if len(matrix) > 1:
                        cls.print_matrix(matrix[-1], '', max_len, start + 1)
                else:
                    # 一维数组
                    if isinstance(length, int):
                        length = [length] * len(matrix)

                    for index, value in enumerate(matrix[:-1]):
                        cls.print_matrix(value, ', ', length[index])
                        
                    cls.print_matrix(matrix[-1], '', length[-1])

            print(']', end=end)
        else:
            # 0维数组
            length -= cls.get_string_length(str(matrix))
            print(' ' * length + str(matrix), end=end)

    # 获取字符串长度
    @staticmethod
    def get_string_length(string):
        return len(str(string).encode('gbk'))
