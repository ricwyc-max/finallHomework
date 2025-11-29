# -*- coding: utf-8 -*-
"""
糖炒栗子计数 —— 褐色 mask + Canny 黑边
边缘以黑色形式叠加进褐色区域，再统一形态学处理
"""
import cv2
import numpy as np
import utils
from math import sqrt,pi



# 1. 读图
img = cv2.imread('lizi.png')
if img is None:
    raise FileNotFoundError('找不到 lizi.png')
utils.show(img, '1. 1_original')
utils.save(img, '1_original.jpg')

# 2. 转 HSV，提取褐色
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
brown_mask = cv2.inRange(hsv, (0, 43, 52), (255, 255, 255))
utils.show(brown_mask, '2. 2_brown_mask')
utils.save(brown_mask, '2_brown_mask.jpg')

# 3. 对灰度图做 Canny，得到白色边缘
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)   # 5×5 核，σ=0 让 OpenCV 自动算
equ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
canny = cv2.Canny(equ, 80, 300)        # 阈值可调
utils.show(canny, '3. Canny')
utils.save(canny, '3.Canny.jpg')


# 5. 开运算去噪
open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
utils.show(opened, '5. 5_opened')
utils.save(opened, '5_opened.jpg')


# 6. 闭运算填小洞
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=3)
utils.show(closed, '6. 6_closed')
utils.save(closed, '6_closed.jpg')




# 7. 二次腐蚀（断连）
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
eroded = cv2.erode(closed, erode_kernel, iterations=9)
utils.show(eroded, '8. 8_eroded')
utils.save(eroded, '8_eroded.jpg')

'''
eroded = cv2.erode(eroded,
                   cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                   iterations=2)
utils.show(eroded, '8. 8_eroded')
utils.save(eroded, '8_eroded.jpg')
'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
canny = cv2.dilate(canny, kernel, iterations=2)
# 4. 把白边→黑边，并叠加到褐色 mask（黑边区域直接=0）
canny_inv = cv2.bitwise_not(canny)          # 白→黑
masked_edge = cv2.bitwise_and(eroded, eroded, mask=canny_inv)
utils.show(masked_edge, '4. 4_masked_edge')
utils.save(masked_edge, '4_masked_edge.jpg')



# 8. 轮廓提取 + 最小外接圆
contours, _ = cv2.findContours(masked_edge, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
print(f'[信息] 原始轮廓数 = {len(contours)}')

final = img.copy()
overlay = cv2.cvtColor(masked_edge, cv2.COLOR_GRAY2BGR)
count = 0
area_list = []

'''
for cnt in contours:
    (x, y), r = cv2.minEnclosingCircle(cnt)
    area_cnt = cv2.contourArea(cnt)
    if r < 10 or r > 60:
        continue
    count += 1
    area_list.append(area_cnt)
    cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.circle(final,   (int(x), int(y)), 3, (0, 0, 255), -1)
'''
# ---------- 合并前，先收集所有圆 ----------
circles = []                                    # 元素：(x, y, r, area)
for cnt in contours:
    (x, y), r = cv2.minEnclosingCircle(cnt)
    area_cnt = cv2.contourArea(cnt)
    if 10 <= r <= 50:                         # 先过一遍半径筛
        circles.append((x, y, r, area_cnt))

# ---------- 重合面积 > 30% 合并 ----------
merged = []
used = [False] * len(circles)

def circle_intersection_area(c1, c2):
    """返回两个圆相交面积（近似解析解），若不相交返回 0"""
    x1, y1, r1, _ = c1
    x2, y2, r2, _ = c2
    d = np.hypot(x2 - x1, y2 - y1)
    if d >= r1 + r2:          # 相离
        return 0.0
    if d <= abs(r1 - r2):     # 内含
        return np.pi * min(r1, r2) ** 2
    # 一般相交
    r1_sq, r2_sq, d_sq = r1 * r1, r2 * r2, d * d
    alpha = np.arccos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
    beta  = np.arccos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
    area = (r1_sq * alpha + r2_sq * beta -
            0.5 * r1_sq * np.sin(2 * alpha) -
            0.5 * r2_sq * np.sin(2 * beta))
    return max(area, 0.0)

def merge_circles(circles):
    """
    不断合并直到不能再合并为止
    circles: 列表，元素为 (cx, cy, r, area)
    返回合并后的列表
    """
    cur = circles.copy()          # 当前圆列表
    merged_any = True             # 上一轮是否合并过

    while merged_any:
        merged_any = False
        n = len(cur)
        # 每一轮从头扫描
        i = 0
        while i < n:
            if merged_any:        # 如果本轮已经合并过，重新扫描
                break
            for j in range(i + 1, n):
                c1, c2 = cur[i], cur[j]
                area_inter = circle_intersection_area(c1, c2)
                ratio = area_inter / (pi * min(c1[2], c2[2]) ** 2)
                if ratio > 0.3:
                    # 合并：中心取中点，半径按面积平方根平均，面积累加
                    new_cx = (c1[0] + c2[0]) / 2
                    new_cy = (c1[1] + c2[1]) / 2
                    new_r  = sqrt((c1[2]**2 + c2[2]**2) )   # 也可按需要改规则
                    new_a  = c1[3] + c2[3]
                    # 删除旧两个，加入新圆
                    if i < j:     # 保证先删大的索引
                        del cur[j], cur[i]
                    else:
                        del cur[i], cur[j]
                    cur.append((new_cx, new_cy, new_r, new_a))
                    merged_any = True
                    break         # 立即重新扫描
            i += 1
    return cur

# 用法
merged = merge_circles(circles)

# ---------- 画合并后的圆 ----------#极大值抑制
count = 0
area_list = []
for (x, y, r, a) in merged:
    count += 1
    area_list.append(a)
    cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.circle(final,   (int(x), int(y)), 3, (0, 0, 255), -1)

# 后续打印结果不变
utils.show(overlay, '9. 9_circle')
utils.save(overlay, '9_circle.jpg')
utils.show(final, '10. 10_final')
utils.save(final, '10_final.jpg')

# 9. 结果
if area_list:
    print(f'检测到栗子个数：{count}')
    print(f'平均单颗像素面积：{np.mean(area_list):.1f} px^2')
else:
    print('未检测到褐色区域，请调整 HSV 或 Canny 阈值！')

cv2.destroyAllWindows()