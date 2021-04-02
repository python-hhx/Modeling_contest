# -*- coding: utf-8 -*-
# @Time    : 2021/4/1 10:49
# @Author  : HHX
# @FileName: logger.py
# @Software: PyCharm

import logging
import os
import time


class Logger:

    def __init__(self, name):
        #创建一个logger
        self.logger = logging.getLogger(name)
        #设置收集的日志等级
        self.logger.setLevel(logging.DEBUG)
        self.now = int(time.time())
        #转换为其它日期的格式，如:"%Y-%m-%d %H:%M:%S"
        self.timeArray = time.localtime(self.now)
        self.otherStyleTime = time.strftime("%Y-%m-%d", self.timeArray)


        #创建一个log文件夹
        if not os.path.exists('./log'):
            os.mkdir('./log')

        #创建一个handler，用于写入日志文件
        logname = os.path.join('./log/', 'log_{}.log'.format(self.otherStyleTime))
        fh = logging.FileHandler(logname, encoding='utf-8')

        #创建一个显示在ter上的handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_log(self):
        return self.logger