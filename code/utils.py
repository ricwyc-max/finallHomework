# -*- coding: utf-8 -*-
"""
小工具：显示与保存
"""
import cv2
import os

def show(img, title='image', wait=0):
    """
    用 OpenCV 弹窗显示图像，按任意键继续
    支持中文路径，但窗口标题可能乱码，不影响使用
    """
    cv2.imshow(title, img)
    cv2.waitKey(wait)          # 0 表示等待按键
    return img

def save(img, fname, root='imgs'):
    """
    保存到 imgs 目录（自动创建）
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, fname)
    cv2.imwrite(path, img)
    print(f'[保存] {path}')


'''

import cv2
import numpy as np
img = cv2.imread('lizi1.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def nothing(x): pass
cv2.namedWindow('track')
cv2.createTrackbar('H_low',  'track', 0, 180, nothing)
cv2.createTrackbar('S_low',  'track', 0, 255, nothing)
cv2.createTrackbar('V_low',  'track', 0, 255, nothing)
cv2.createTrackbar('H_high', 'track', 0, 255, nothing)
cv2.createTrackbar('S_high', 'track', 0, 255, nothing)
cv2.createTrackbar('V_high', 'track', 0, 255, nothing)


while True:
    h_l = cv2.getTrackbarPos('H_low',  'track')
    s_l = cv2.getTrackbarPos('S_low',  'track')
    v_l = cv2.getTrackbarPos('V_low',  'track')
    h_h = cv2.getTrackbarPos('H_high', 'track')
    s_h = cv2.getTrackbarPos('S_high', 'track')
    v_h = cv2.getTrackbarPos('V_high', 'track')


    mask = cv2.inRange(hsv, (h_l, s_l, v_l), (h_h, s_h, v_h))
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == 27: break   # ESC 退出
cv2.destroyAllWindows()
'''