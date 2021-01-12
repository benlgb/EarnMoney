# -*- coding: utf-8 -*-

import re
import os
import sys
import time
import pickle
import base64
import urllib3
import requests
import numpy as np

from cv2 import cv2
from getopt import getopt
from bs4 import BeautifulSoup
from itertools import product

from src.game import BaseGame, BaseGameException

urllib3.disable_warnings()


class NextLevel(BaseGameException):
    pass


class Letter:
    def __init__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs

    def __getattr__(self, attr):
        return self.kwargs[attr]

    def __str__(self):
        return str(self.text)

    def __repr__(self):
        return 'Letter(%s)' % str(self.text)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.text == other
        return super().__eq__(other)


class ChengYuHongBaoQun(BaseGame):
    # 文件路径
    SOURCE_DIR = 'chengyuhongbaoqun'
    LETTERS_DIR = os.path.join(SOURCE_DIR, 'letters')
    UNSIGN_LETTERS_DIR = os.path.join(SOURCE_DIR, 'unsign_letters')
    SUPPLED_IDIOMS = os.path.join(SOURCE_DIR, 'idioms.txt')
    CACHE_REQUEST_IDIOMS = os.path.join(SOURCE_DIR, 'request_idioms.cache')

    # 文字
    BG_COLOR = {
        'yellow': [
            np.array((20, 127, 127), np.uint8),
            np.array((22, 255, 255), np.uint8)
        ],
        'gray': [
            np.array((10, 10, 160), np.uint8),
            np.array((30, 40, 180), np.uint8)
        ],
        'white': [
            np.array((0, 0, 210), np.uint8),
            np.array((180, 12, 230), np.uint8)
        ],
        'green': [
            np.array((39, 127, 127), np.uint8),
            np.array((41, 255, 255), np.uint8)
        ]
    }

    LETTER_COLOR = {
        'black': [
            np.array((0, 0, 0), np.uint8),
            np.array((180, 255, 10), np.uint8)
        ],
        'red': [
            np.array((5, 127, 127), np.uint8),
            np.array((7, 255, 255), np.uint8)
        ]
    }

    LETTER_TYPE_PENDING = 0
    LETTER_TYPE_EMPTY = 1
    LETTER_TYPE_ALREADY = 2
    LETTER_TYPE_FINISHED = 3
    LETTER_TYPE_UNEXCEPTED = -1
    LETTER_TYPE_ERROR = -2
    LETTER_TYPE_UNCHECK = -3

    LETTER_TYPES_ERROR = [
        LETTER_TYPE_ERROR,
        LETTER_TYPE_UNCHECK
    ]

    LETTER_TYPES_PLAYING = [
        LETTER_TYPE_ALREADY,
        LETTER_TYPE_FINISHED
    ]

    # 成语查询
    DEFAULT_HEADERS = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    POST_HEADERS = {
        'content-type': 'application/x-www-form-urlencoded'
    }
    POST_HEADERS.update(DEFAULT_HEADERS)
    IDIOM_URL_1 = 'https://chengyu.duwenz.com/manage/javaajax_cy.aspx'
    IDIOM_URL_2 = 'https://www.chengyuwang.com/chaxun.php'

    # 百度OCR
    API_KEY = 'wVsYLHt7umu0YRDigdl2aTAp'
    SECRET_KEY = 'SEpXFO3htCv6MG06ADdo1wevfnoNaEl3'
    ACCESS_TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
    ACCESS_TOKEN = '24.aa758a80cc2e4721d65fb0e5bca95f72.2592000.1611555126.282335-23394611'
    GENERAL_OCR_URL = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic'
    GENERAL_WITH_POSITION_OCR_URL = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 配置
        self.screencap_interval = 3
        self.access_token = self.ACCESS_TOKEN
        self.half = int(self.size[1] * self.scale / 2)

        # 初始化信息
        self.pos = []
        self.pending = []
        self.ad_count = 0
        self.color_0_0 = None

        # 缓存
        self.suppled_idioms = []
        self.corrected_letter_texts = self.load_letter_imgs()
        self.UNSIGN_LETTERS_DIR = os.path.join(self.UNSIGN_LETTERS_DIR, self.device[-5:])

        if not os.path.isdir(self.UNSIGN_LETTERS_DIR):
            os.makedirs(self.UNSIGN_LETTERS_DIR)

        with open(self.SUPPLED_IDIOMS, encoding='utf-8') as f:
            for idiom in f.readlines():
                idiom = idiom.strip()

                if re.match(r'^[\u4e00-\u9fa5]{4}$', idiom):
                    self.suppled_idioms.append(idiom)

        # 图片
        self.imgs = {
            'next': self.imread('next.png'),
            'goon': self.imread('goon.png'),
            'goon_1': self.imread('goon_1.png'),
            'start': self.imread('start.png'),
            'getit': self.imread('getit.png'),
            'return': self.imread('return.png'),
            'cancel': self.imread('cancel.png'),
            'cancel_1': self.imread('cancel_1.png'),
            'redpack': self.imread('redpack.png'),
            'close_1': self.imread('close_1.png'),
            'close_2': self.imread('close_2.png'),
            'close_3': self.imread('close_3.png'),
            'close_4': self.imread('close_4.png'),
            'bin_close_1': self.imread('bin_close_1.png', 0),
            'bin_close_2': self.imread('bin_close_2.png', 0),
            'bin_close_3': self.imread('bin_close_3.png', 0),
            'bin_close_4': self.imread('bin_close_4.png', 0),
            'bin_close_5': self.imread('bin_close_5.png', 0),
            'bin_close_6': self.imread('bin_close_6.png', 0),
            'bin_close_7': self.imread('bin_close_7.png', 0)
        }
        print(self.size)
        self.btns = [
            (self.imgs['close_1'], slice(self.half)),
            (self.imgs['bin_close_5'], slice(self.half), True),
            (self.imgs['bin_close_6'], slice(self.half), True),
            (self.imgs['close_2'], slice(self.half)),
            self.imgs['getit'],
            (self.imgs['close_3'], slice(self.half)),
            self.imgs['close_4'],
            (self.imgs['return'], slice(300)),
            self.imgs['start'],
            self.imgs['cancel'],
            self.imgs['cancel_1'],
            self.imgs['goon_1'],
            (self.imgs['bin_close_1'], slice(200), True),
            (self.imgs['bin_close_2'], slice(200), True),
            (self.imgs['bin_close_3'], slice(200), True),
            (self.imgs['bin_close_4'], slice(200), True),
            (self.imgs['bin_close_7'], slice(200), True),
        ]
    
    def run(self):
        count = 0

        while True:
            try:
                bmp = self.screencap()
            except NextLevel:
                continue
            
            if self.remove_error_letters(bmp):
                bmp = self.screencap()

            self.initialization(bmp)

            try:
                if self.fill_exact_idioms():
                    count = 0
                else:
                    count += 1

                if count >= 2 and self.fill_possible_idioms():
                    count -= 1
            except NextLevel:
                pass

            print('\n[+] finished pos:')
            self.print_matrix(self.pos)

            if count > 2 and len(self.pending) > 0:
                print('[-] finished pending:', self.pending)
                raise Exception()

            print()

    # 填充可能的成语
    # 部分缺少的文字能在待选文字中找到
    def fill_possible_idioms(self):
        info = []

        for idiom in self.find_all_idioms():
            req_idioms = self.request_for_idioms(idiom)
            req_idioms += self.find_all_suppled_idioms(idiom)

            for req_idiom in req_idioms:
                count = 0
                letters = []
                empty_pts = []
                temp_pending = self.pending[:]

                for (status, pt), text in zip(idiom, req_idiom):
                    if status != -1:
                        continue

                    if text in temp_pending:
                        _index = temp_pending.index(text)
                        letter = temp_pending.pop(_index)
                        letters.append((letter, pt))
                    else:
                        count += 1
                        empty_pts.append(pt)
            
                if len(letters) == 0 or count == 0:
                    continue

                info.append((count, letters, empty_pts, temp_pending))

            empty_pts = []

            for status, pt in idiom:
                if status != -1:
                    continue

                empty_pts.append(pt)
            
            info.append((len(empty_pts), [], empty_pts, self.pending[:]))
            
        info.sort(key=lambda i: i[0])
        
        for count, letters, empty_pts, temp_pending in info:
            for letter, pt in letters:
                self.tap(pt)
                self.tap(letter.pt)

            possible_letters = self.iter_possible_letters(empty_pts, temp_pending)

            if possible_letters is not None:
                for letter, pt in letters + possible_letters:
                    self.set_pos(pt, letter)
                    self.pending.remove(letter)
                return True

            for _, pt in letters:
                self.tap(pt)

        return False

    # 遍历可能的文字
    def iter_possible_letters(self, pts, letters):
        self.tap(pts[0])

        for letter in letters:
            self.tap(letter.pt)

            if len(pts) == 1:
                if self.check_finished_letter(pts[0]):
                    time.sleep(5)
                    self.screencap()
                    return [(letter, pts[0])]
            else:
                temp_letters = letters[:]
                temp_letters.remove(letter)
                result = self.iter_possible_letters(pts[1:], temp_letters)

                if result is not None:
                    return [(letter, pts[0])] + result

                self.tap(pts[0])

        if len(pts) == 1:
            self.tap(pts[0])
        
        return None

    # 填充确定的成语
    # 所有缺少的文字都能在待选文字中找到
    def fill_exact_idioms(self):
        is_finished = False
        pending = self.pending
        idioms = self.find_all_idioms()

        while True:
            try:
                idiom = next(idioms)
            except StopIteration:
                break

            req_idioms = self.request_for_idioms(idiom)
            req_idioms += self.find_all_suppled_idioms(idiom)

            for req_idiom in req_idioms:
                pts = []

                for (status, pt), text in zip(idiom, req_idiom):
                    if status != -1:
                        continue

                    if text not in pending:
                        break

                    index = pending.index(text)
                    letter = pending.pop(index)
                    pts.append((letter, pt))
                else:
                    print('\n[+] trying idiom:', req_idiom)

                    for letter, pt in pts:
                        self.tap(pt)
                        self.tap(letter.pt)

                    if self.check_finished_letter(pts[0][1]):
                        for letter, pt in pts:
                            self.set_pos(pt, letter)

                        idioms = self.find_all_idioms()

                        # if self.ad_count < 20:
                        #     time.sleep(5)
                            
                        #     if self.screencap(True):
                        #         self.ad_count = 0
                        #     else:
                        #         self.ad_count += 1

                        is_finished = True
                        break

                    for _, pt in pts:
                        self.tap(pt)

                for letter, _ in pts:
                    pending.append(letter)

        return is_finished

    # 寻找所有成语位置
    def find_all_idioms(self):
        pos = self.pos
        axis_x = range(1, len(pos[0]) - 1)
        axis_y = range(1, len(pos) - 1)

        for j, i in product(axis_y, axis_x):
            if pos[j][i] == 0:
                continue

            if i == 1 or pos[j][i-1] == 0:
                if i < len(pos[0]) - 3 and pos[j][i+1] != 0:
                    idiom = (
                        (pos[j][i], (pos[0][i], pos[j][0])),
                        (pos[j][i+1], (pos[0][i+1], pos[j][0])),
                        (pos[j][i+2], (pos[0][i+2], pos[j][0])),
                        (pos[j][i+3], (pos[0][i+3], pos[j][0]))
                    )

                    if self.check_pre_idiom(idiom):
                        yield idiom
            
            if j == 1 or pos[j-1][i] == 0:
                if j < len(pos) - 3 and pos[j+1][i] != 0:
                    idiom = (
                        (pos[j][i], (pos[0][i], pos[j][0])),
                        (pos[j+1][i], (pos[0][i], pos[j+1][0])),
                        (pos[j+2][i], (pos[0][i], pos[j+2][0])),
                        (pos[j+3][i], (pos[0][i], pos[j+3][0]))
                    )

                    if self.check_pre_idiom(idiom):
                        yield idiom

    # 判断是否为待寻找成语
    def check_pre_idiom(self, idiom):
        count = 0

        for status, _ in idiom:
            if status == -1:
                count += 1
            elif status == 0:
                return False

        return 0 < count and count < 4

    # 寻找所有可能的补充成语
    def find_all_suppled_idioms(self, idiom):
        idioms = []

        for _idiom in self.suppled_idioms:
            for (letter, _), text in zip(idiom, _idiom):
                if letter == -1:
                    continue
                elif letter.text != text:
                    break
            else:
                idioms.append(_idiom)
        
        return idioms

    # 请求所有可能的成语
    def request_for_idioms(self, idiom):
        idiom_text = []

        for letter, _ in idiom:
            if isinstance(letter, Letter) and letter.text is not None:
                idiom_text.append(letter.text)
            else:
                idiom_text.append(-1)

        idiom_text = tuple(idiom_text)
        cache_request_idioms = {}

        if os.path.isfile(self.CACHE_REQUEST_IDIOMS):
            while True:
                try:
                    with open(self.CACHE_REQUEST_IDIOMS, 'rb') as f:
                        cache_request_idioms = pickle.load(f)
                    break
                except Exception:
                    pass

        if idiom_text in cache_request_idioms:
            return cache_request_idioms[idiom_text]

        idioms = set()

        # 第一个成语网站
        params = {'a': 'getcysearch'}

        for index, text in enumerate(idiom_text):
            if text != -1:
                params['k%d' % (index + 1)] = text
            else:
                params['k%d' % (index + 1)] = ''

        res = requests.post(self.IDIOM_URL_1, data=params, headers=self.POST_HEADERS)
        soup = BeautifulSoup(res.text, 'lxml')

        for td in soup.find_all('td', 'cy'):
            if len(td.text) == 4:
                idioms.add(td.text)

        while True:
            res = requests.get(self.IDIOM_URL_2, params=params, headers=self.DEFAULT_HEADERS, verify=False)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'lxml')

            for a in soup.dd.find_all('a'):
                if len(a.text) == 4:
                    idioms.add(a.text)

            if not soup.find('a', title='下一页'):
                break

            params['p'] += 1

        idioms = list(idioms)
        print('\n[+] save idioms', str(idiom_text) + ':', idioms)
        cache_request_idioms[idiom_text] = idioms

        while True:
            try:
                with open(self.CACHE_REQUEST_IDIOMS, 'wb') as f:
                    pickle.dump(cache_request_idioms, f)
                break
            except Exception:
                pass

        return idioms

    # 初始化pos与pending
    def initialization(self, bmp):
        print()

        # 重置
        self.pending = []

        if len(self.pos) < 11 or len(self.pos[0]) < 11:
            self.pos = np.zeros((2, 2), int)
            self.pos[0, 1] = self.size[0] + 1
            self.pos[1, 0] = self.size[1] + 1
        else:
            for j in range(1, len(self.pos) - 1):
                for i in range(1, len(self.pos[0]) - 1):
                    self.pos[j][i] = 0

        # 初始赋值
        playing = []

        for letter in self.find_all_letters(bmp):
            if letter.type == self.LETTER_TYPE_PENDING:
                self.pending.append(letter)
            elif letter.type == self.LETTER_TYPE_EMPTY:
                self.set_pos(letter.pt, -1)
            elif letter.type in self.LETTER_TYPES_PLAYING:
                playing.append(letter)
                self.set_pos(letter.pt, 1)

        self.pending.reverse()

        if isinstance(self.pos, np.ndarray):
            self.pos = self.pos.tolist()

        for letter in playing:
            self.set_pos(letter.pt, letter)

        # 请求百度OCR接口
        letters = []

        for letter in playing + self.pending:
            if letter.text is None:
                letters.append(letter)

        if len(letters) > 0:
            print('[+] 向百度OCR请求%d个文字' % len(letters))
            img = self.combine_letter_imgs(letters)
            texts = self.request_baidu_ocr(img)

            if len(texts) == len(letters):
                for letter, text in zip(letters, texts):
                    letter.text = text
            else:
                for info in self.request_baidu_ocr_with_position(img):
                    location = info['location']
                    x1 = location['left']
                    x2 = x1 + location['width']

                    for letter in letters:
                        if x1 < letter.cx and letter.cx < x2:
                            letter.text = info['char']
                            break

            for filename in os.listdir(self.UNSIGN_LETTERS_DIR):
                os.remove(os.path.join(self.UNSIGN_LETTERS_DIR, filename))

            for letter in playing + self.pending:
                self.save_letter_img(letter.text, letter.img)

        print('[+] pre pos:')
        self.print_matrix(self.pos)
        print('[+] pre pending:', self.pending)

    # 合并文字图片
    def combine_letter_imgs(self, letters, padding=20, letter_space=0):
        height = 0
        width = padding * 2 - letter_space

        for letter in letters:
            h, w = letter.img.shape[:2]
            width += w + letter_space
            height = max(height, h)

        height += padding * 2
        x = int(padding - letter_space / 2)
        img = np.zeros((height, width), np.uint8)
        img.fill(255)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for letter in letters:
            h, w = letter.img.shape[:2]
            y = int((height - h) / 2)
            img[y:y+h, x:x+w] = letter.img
            letter.cx = int(x + w / 2)
            x += w + letter_space

        return img

    # 请求百度文字识别
    def request_baidu_ocr(self, img):
        params = '?access_token=' + self.access_token
        image = cv2.imencode('.jpg', img)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        res = requests.post(self.GENERAL_OCR_URL + params, data={
                'image': image_code
            }, headers=self.POST_HEADERS)

        try:
            return res.json()['words_result'][0]['words']
        except (IndexError, KeyError):
            return ''

    # 请求百度文字识别（标准含位置版）
    def request_baidu_ocr_with_position(self, img):
        params = '?access_token=' + self.access_token
        image = cv2.imencode('.jpg', img)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        res = requests.post(self.GENERAL_WITH_POSITION_OCR_URL + params, data={
                'image': image_code,
                'recognize_granularity': 'small'
            }, headers=self.POST_HEADERS)

        try:
            return res.json()['words_result'][0]['chars']
        except (IndexError, KeyError):
            return []

    # 设置pos
    def set_pos(self, pt, letter, r=5):
        i = j = 0

        for i, x in enumerate(self.pos[0]):
            if pt[0] > x + r:
                continue

            if isinstance(self.pos, np.ndarray) and pt[0] < x - r:
                self.pos = np.insert(self.pos, i, 0, 1)
                self.pos[0, i] = pt[0]

            break

        for j, y in enumerate(self.pos):
            if pt[1] > y[0] + r:
                continue

            if isinstance(self.pos, np.ndarray) and pt[1] < y[0] - r:
                self.pos = np.insert(self.pos, j, 0, 0)
                self.pos[j, 0] = pt[1]

            break

        self.pos[j][i] = letter

    # 检查文字是否已完成
    def check_finished_letter(self, pt, r=34):
        x, y = pt
        bmp = self.screencap()
        img = bmp[y-r:y+r, x-r:x+r]
        _type = self.get_letter_type(img=img)
        print('[+] letter type:', _type)
        return _type == self.LETTER_TYPE_FINISHED

    # 移除所有错误文字
    def remove_error_letters(self, bmp):
        is_remove = False

        for letter in self.find_all_letters(bmp):
            if letter.type in self.LETTER_TYPES_ERROR:
                print('[-] remove error letter:', letter.pt)
                self.tap(letter.pt)
                is_remove = True

        return is_remove

    # 寻找所有的文字
    def find_all_letters(self, bmp, r=3):
        bmp_hsv = cv2.cvtColor(bmp[:-200], cv2.COLOR_BGR2HSV)

        for key, colors in self.BG_COLOR.items():
            temp = cv2.inRange(bmp_hsv, *colors)
            contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if w < 60 or w > 80 or h < 60 or h > 80:
                    continue

                img = bmp[y+r:y+h-r, x+r:x+w-r]
                img_hsv = bmp_hsv[y:y+h, x:x+w]
                pt = int(x + w / 2), int(y + h / 2)
                text = self.get_letter_text(img)
                _type = self.get_letter_type(img_hsv, key)
                yield Letter(text, img=img, pt=pt, type=_type)

    # 获取文字类型
    def get_letter_type(self, img_hsv=None, bgColor=None, img=None):
        if img_hsv is None:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if bgColor is None:
            for key, colors in self.BG_COLOR.items():
                temp = cv2.inRange(img_hsv, *colors)

                if np.sum(temp == 255) / img_hsv.size > 0.05:
                    bgColor = key
                    break

        if bgColor == 'yellow':
            temp = cv2.inRange(img_hsv, *self.LETTER_COLOR['black'])
            
            if np.sum(temp == 255) / img_hsv.size > 0.005:
                return self.LETTER_TYPE_PENDING
            else:
                return self.LETTER_TYPE_ERROR
        elif bgColor == 'gray':
            temp = cv2.inRange(img_hsv, *self.LETTER_COLOR['black'])

            if np.sum(temp == 255) / img_hsv.size > 0.005:
                return self.LETTER_TYPE_UNCHECK
            else:
                return self.LETTER_TYPE_EMPTY
        elif bgColor == 'white':
            return self.LETTER_TYPE_ALREADY
        elif bgColor == 'green':
            return self.LETTER_TYPE_FINISHED

        return self.LETTER_TYPE_UNEXCEPTED

    # 获取文字文本
    def get_letter_text(self, img):
        max_conf = 0
        max_text = None

        for text, text_img in self.corrected_letter_texts:
            match = self.find_template(img, text_img, 0.98)

            if match is not None and match['conf'] > max_conf:
                max_conf = match['conf']
                max_text = text

        return max_text

    # 加载文字图片
    def load_letter_imgs(self):
        imgs = []

        for filename in os.listdir(self.LETTERS_DIR):
            match = re.match(r'^letter_(.)(_\d+)?.png$', filename)
            
            if match is None:
                continue

            path = os.path.join(self.LETTERS_DIR, filename)
            img = self.imread(path, join=False)
            imgs.append((match.group(1), img))

        return imgs

    # 保存文字图片
    def save_letter_img(self, text, img, p=3, min_width=40, min_height=40):
        count = 0
        filename = 'letter_%s.png' % text
        path = os.path.join(self.UNSIGN_LETTERS_DIR, filename)

        while os.path.isfile(path):
            count += 1
            filename = 'letter_%s_%d.png' % (text, count)
            path = os.path.join(self.UNSIGN_LETTERS_DIR, filename)

        temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, temp_bin = cv2.threshold(255 - temp, 200, 255, cv2.THRESH_BINARY)

        if (temp_bin == 0).all():
            _, temp_bin = cv2.threshold(temp, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(temp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, x2 = temp.shape[1], 0
        y1, y2 = temp.shape[0], 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = min(x1, x)
            x2 = max(x2, x + w)
            y1 = min(y1, y)
            y2 = max(y2, y + h)

        if x2 - x1 < min_width:
            x1 = int((x1 - x2 + min_width) / 2)
            x2 = x1 + min_width

        if y2 - y1 < min_height:
            y1 = int((y1 - y2 + min_height) / 2)
            y2 = y1 + min_height

        x1 = max(0, x1 - p)
        y1 = max(0, y1 - p)
        x2 = min(img.shape[1], x2 + p)
        y2 = min(img.shape[0], y2 + p)
        self.imwrite(path, img[y1:y2, x1:x2])

    # 读取图片
    def imread(self, path, flags=1, join=True):
        if join:
            path = os.path.join(self.SOURCE_DIR, path)

        return super().imread(path, flags)

    # 点击
    def tap(self, pt, sleep=0):
        super().tap(pt)
        time.sleep(sleep)

    # 点击按钮
    def tap_btn(self, btn, bmp, threshold=0.8):
        match = self.find_template(bmp, btn, threshold)

        if match is not None:
            self.tap(match['pt'], 3)
            return True

        return False

    # 截屏
    def screencap(self, check=False):
        bmp = None
        count = 0

        while True:
            count += 1
            bmp = super().screencap()

            # 参与互动
            match = self.find_template(bmp, self.imgs['goon'])
            
            if match is not None:
                x1, y1 = match['rect'][0]
                x2, y2 = match['rect'][1]
                temp = bmp[y1:y2, x1:x2]
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                _, temp = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY)

                if (temp == 0).all():
                    self.tap(match['pt'])
                    continue

            # 点击按钮
            for btn in self.btns:
                temp = bmp
                is_gray = False
                match_threshold = 0.8

                if isinstance(btn, tuple):
                    if len(btn) >= 2 and btn[1] is not None:
                        temp = temp[btn[1]]
                    if len(btn) >= 3 and btn[2] is not None:
                        is_gray = btn[2]
                    if len(btn) >= 4 and btn[3] is not None:
                        match_threshold = btn[3]
                    btn = btn[0]
                
                # self.imwrite('test.png', temp)
                # self.showImg(temp)
                
                if is_gray:
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

                    for threshold in [160, 180, 200]:
                        _, temp_bin = cv2.threshold(temp, threshold, 255, cv2.THRESH_BINARY)

                        # self.imwrite('test.png', temp_bin)
                        # self.showImg(temp_bin)

                        if self.tap_btn(btn, temp_bin, match_threshold):
                            break
                    else:
                        continue
                elif not self.tap_btn(btn, temp, match_threshold):
                    continue
                break
            else:
                # 下一关
                if self.tap_btn(self.imgs['next'], bmp):
                    raise NextLevel()

                # 红包
                if self.find_template(bmp[:200], self.imgs['redpack']):
                    continue

                if self.color_0_0 is None:
                    self.color_0_0 = bmp[0, 0]
                    break
                elif (bmp[0, 0] == self.color_0_0).all():
                    break

                self.tap('back', 3)

        if check:
            return count > 1

        return bmp


if __name__ == '__main__':
    device = '127.0.0.1:62030'
    opts, _ = getopt(sys.argv[1:], '-d:', ['device='])

    for name, value in opts:
        if name in ('-d', '--device'):
            device = value

    game = ChengYuHongBaoQun(device)
    game.run()
