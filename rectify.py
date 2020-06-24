# author: kxie
# mail: xiekai@sundear.com
# date: 2020-06-21

"""
software: id-card rectification
version: 1.3.0
"""

import sys
import cv2
import imutils
import numpy as np
from skimage import exposure
from skimage import img_as_ubyte
from imutils.perspective import four_point_transform
from itertools import combinations


# 计算两条直线交点
def crossPoint(line1, line2):
    x = 0
    y = 0
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0

    return [x, y]


# 裁剪身份证圆角
def setCorner(img, r):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    row = img.shape[0]
    col = img.shape[1]

    for i in range(0, r):
        for j in range(0, r):
            if (r - i) * (r - i) + (r - j) * (r - j) > r * r:
                alpha_channel[i][j] = 0

    for i in range(0, r):
        for j in range(col - r, col):
            if (r - i) * (r - i) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                alpha_channel[i][j] = 0

    for i in range(row - r, row):
        for j in range(0, r):
            if (r - row + i + 1) * (r - row + i + 1) + (r - j) * (r - j) > r * r:
                alpha_channel[i][j] = 0

    for i in range(row - r, row):
        for j in range(col - r, col):
            if (r - row + i + 1) * (r - row + i + 1) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                alpha_channel[i][j] = 0

    img_bgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return img_bgra


# 检测边缘
def getOutline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    meanG = cv2.mean(gray)
    light_TH = 150
    if meanG[0] > light_TH:
        gray = exposure.adjust_gamma(gray, gamma=6)
        gray = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.02)
        gray = img_as_ubyte(gray)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.medianBlur(closing, 5)
    blurred = cv2.bilateralFilter(blurred, d=0, sigmaColor=15, sigmaSpace=10)

    edged = cv2.Canny(blurred, 75, 200)

    # debug
    # cv2.imwrite("01gray.png", gray)
    # cv2.imwrite("02closing.png", closing)
    # cv2.imwrite("03blurred.png", blurred)
    # cv2.imwrite("04edged.png", edged)

    return img, gray, edged


# 获取身份证四个角点
def getCnt(edged, img, ratio):
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
    c = cnts[0]

    edgelines = np.zeros(edged.shape, np.uint8)
    cv2.drawContours(edgelines, [c], 0, (255, 255, 255), 2)
    lines = cv2.HoughLines(edgelines, 1, np.pi / 180, 200)
    strong_lines = np.zeros([4, 1, 2])
    n2 = 0
    for n1 in range(0, len(lines)):
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=40)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1

    lines1 = np.zeros((len(strong_lines), 4), dtype=int)
    for i in range(0, len(strong_lines)):
        rho, theta = strong_lines[i][0][0], strong_lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        lines1[i][0] = int(x0 + 1000 * (-b))
        lines1[i][1] = int(y0 + 1000 * (a))
        lines1[i][2] = int(x0 - 1000 * (-b))
        lines1[i][3] = int(y0 - 1000 * (a))

    approx = np.zeros((len(strong_lines), 1, 2), dtype=int)
    index = 0
    combs = list((combinations(lines1, 2)))
    for twoLines in combs:
        x1, y1, x2, y2 = twoLines[0]
        x3, y3, x4, y4 = twoLines[1]
        [x, y] = crossPoint([x1, y1, x2, y2], [x3, y3, x4, y4])
        if 0 < x < img.shape[1] and 0 < y < img.shape[0] and index < 4:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 3)
            approx[index] = (int(x), int(y))
            index = index + 1

    return approx * ratio


# 结果微调
def fineTune(img):
    offset = int(2 * ratio)
    img = img[offset + 15:img.shape[0] - offset,
                 int(offset * 2):img.shape[1] - int(offset * 2), :]
    img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 856 * 540)))
    img = setCorner(img, r)
    return img


if __name__ == "__main__":
    rectify, input_path, output_path = sys.argv
    image = cv2.imread(input_path)

    limit = 1000
    img = cv2.resize(image, (limit, int(limit * image.shape[0] / image.shape[1])))
    ratio = image.shape[1] / limit
    r = int(25 * ratio)

    img, gray, edged = getOutline(img)
    corners = getCnt(edged, img, ratio)
    result = four_point_transform(image, corners.reshape(4, 2))
    result = fineTune(result)

    cv2.imwrite(output_path, result)
    print("success.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()