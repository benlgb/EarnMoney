# -*- coding: utf-8 -*-

import os
import re
import time
import uuid
import aircv
import random
from cv2 import cv2


class ADBBaseException(Exception):
    pass


class NoMatchException(ADBBaseException):
    pass


class ConnectionError(ADBBaseException):
    pass


class ADB:
    default_sd_dir = '/sdcard/Python/'
    default_screencap_dir = './screencap'

    def __init__(self, device, adb='adb'):
        self.device = device
        self.adb = adb
        self.tap_time = 0
        self.tap_interval = 0
        self.uid = uuid.uuid1().hex[:8]

    def run_cmd(self, cmd, result=False, show_cmd=True):
        cmd = '%s -s %s %s' % (self.adb, self.device, cmd)

        if show_cmd:
            print('[+] run command: %s' % cmd)

        if result:
            return os.popen(cmd).read()
        else:
            os.system(cmd)
            return None

    # 获取屏幕大小
    def get_size(self):
        result = self.run_cmd('shell wm size', result=True)
        match = re.search(r'(\d+)x(\d+)', result)

        if match is None:
            raise ConnectionError()
        else:
            return tuple(map(int, match.groups()))

    # 截屏
    def screencap(self, filename=None, sd_dir=None, screencap_dir=None):
        if sd_dir is None:
            sd_dir = self.default_sd_dir

        if screencap_dir is None:
            screencap_dir = self.default_screencap_dir

        if not os.path.isdir(screencap_dir):
            os.makedirs(screencap_dir)

        if filename is None:
            index = 0
            while True:
                filename = 'screencap_%d.png' % index
                path = os.path.join(screencap_dir, filename)

                if not os.path.isfile(path):
                    break
            
        sd_path = os.path.join(sd_dir, filename)
        self.run_cmd('shell screencap -p %s' % sd_path)
        self.run_cmd('pull %s %s' % (sd_path, screencap_dir), show_cmd=False)
        return os.path.join(screencap_dir, filename)

    # 键盘事件
    def keyevent(self, keycode):
        if keycode == 'menu':
            keycode = 82
        elif keycode == 'back':
            keycode = 4

        self.run_cmd('shell input keyevent %d' % keycode)
        return keycode

    # 点击事件
    def tap(self, pt, interval=None):
        if interval is None:
            interval = self.tap_interval

        while time.time() - self.tap_time < interval:
            time.sleep(0.1)

        if isinstance(pt, (int, str)):
            return self.keyevent(pt)

        pt = (int(pt[0]), int(pt[1]))
        self.run_cmd('shell input tap %d %d' % pt)
        self.tap_time = time.time()
        return pt

    # 点击按钮
    def tap_btn(self, btn, bmp=None, threshold=0.9, ignore_error=False):
        if bmp is None:
            path = self.screencap('btn_%s.png' % self.uid)
            bmp = cv2.imread(path)

        btn_pt = None
        pts = aircv.find_all_template(bmp, btn, threshold=threshold)

        if pts is not None and len(pts) > 0:
            btn_pt = pts[0]['result']
        elif ignore_error:
            return None
        else:
            raise NoMatchException('could not match the button')

        return self.tap(btn_pt)

    # 滑动事件
    def swipe(self, pt1, pt2, duration=0):
        cmd = 'shell input swipe %d %d %d %d' % (pt1 + pt2)

        if duration > 0:
            cmd += ' %d' % duration

        self.run_cmd(cmd)
        