# author: kxie
# mail: xiekai@sundear.com
# date: 2020-07-18

"""
software: id-card rectification
version: 2.0.0
"""


import sys
import cv2
import torch
import imutils
import numpy as np
import scipy.misc as m
from skimage import exposure
from skimage import img_as_ubyte
from collections import OrderedDict
from imutils.perspective import four_point_transform
from itertools import combinations
from os.path import join as pjoin

from models import get_model
from torchvision import transforms


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_card_colormap():
    return np.asarray([[0, 0, 0], [255, 255, 255]])

def decode_segmap(label_mask):
    label_colors = get_card_colormap()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 2):
        r[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        b[label_mask == ll] = label_colors[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb.astype(np.uint8)


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


def getOutlineWithCNN(img, model_path, limit):

    # load model
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    model = get_model({'arch': 'UNetRNN'}, n_classes=2).to(device)
    state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # img2tensor
    image = cv2.resize(img, (1000, int(1000 * img.shape[0] / img.shape[1])), interpolation = cv2.INTER_LINEAR)
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.196, 0.179, 0.323], [0.257, 0.257, 0.401])
        ]
    )
    image = tf(image)
    image = image.unsqueeze(0)

    # get result
    with torch.no_grad():
        img_val = image.to(device)
        res = model(img_val)
        pred = np.squeeze(res.data.max(1)[1].cpu().numpy())
        edged = decode_segmap(pred)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
        edged = cv2.resize(edged, (limit, int(limit * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_NEAREST)

        # m.imsave("results/" + name + "_edge.png", edged)

    return edged


# 获取身份证四个角点
def getCnt(edged, img, ratio):
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    mask = np.zeros((edged.shape[0], edged.shape[1]), np.uint8)
    mask[10:edged.shape[0]-10, 10:edged.shape[1]-10] = 1
    edged = edged * mask

    # cv2.imwrite("result/" + "card1_edge.png", edged)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)

    edgelines = np.zeros(edged.shape, np.uint8)
    cNum = 4

    for i in range(min(cNum, len(cnts))):
        TH = 1/20.0
        # print(cv2.contourArea(cnts[i]))
        # print(img.shape[0] * img.shape[1])
        if cv2.contourArea(cnts[i]) < TH * img.shape[0] * img.shape[1]:
            cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)
        else:
            cv2.drawContours(edgelines, [cnts[i]], 0, (1, 1, 1), -1)
            edgelines = edgelines * edged
            break
        # cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)

    # cv2.imwrite("results/" + name + "_edgelines.png", edgelines)

    lines = cv2.HoughLines(edgelines, 1, np.pi / 180, 200)

    if lines is None or len(lines) < 4:
        return

    # draw lines (debug)
    # lines_draw = np.zeros((len(lines), 4), dtype=int)
    # img_draw = img.copy()
    # for i in range(0, len(lines)):
    #     rho, theta = lines[i][0][0], lines[i][0][1]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     lines_draw[i][0] = int(x0 + 1000 * (-b))
    #     lines_draw[i][1] = int(y0 + 1000 * (a))
    #     lines_draw[i][2] = int(x0 - 1000 * (-b))
    #     lines_draw[i][3] = int(y0 - 1000 * (a))
    #     cv2.line(img_draw, (lines_draw[i][0], lines_draw[i][1]), (lines_draw[i][2], lines_draw[i][3]), (0, 255, 0), 1)

    strong_lines = np.zeros([4, 1, 2])
    n2 = 0

    if lines is None:
        return

    for n1 in range(0, len(lines)):
        if n2 == 4:
            break
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                c1 = np.isclose(abs(rho), abs(strong_lines[0:n2, 0, 0]), atol=80)
                c2 = np.isclose(np.pi - theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                c = np.all([c1, c2], axis=0)
                if any(c):
                    continue
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=40)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4 and theta!=0:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1

    # draw strong lines
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
    model_path = "model.pkl"
    image = cv2.imread(input_path)

    limit = 1000
    img = cv2.resize(image, (limit, int(limit * image.shape[0] / image.shape[1])))
    ratio = image.shape[1] / limit
    r = int(25 * ratio)

    try:
        # img, gray, edged = getOutline(img)
        edged = getOutlineWithCNN(image, model_path, limit)
        corners = getCnt(edged, img, ratio)
        result = four_point_transform(image, corners.reshape(4, 2))
        result = fineTune(result)
        cv2.imwrite(output_path, result)
        print("success.")
    except:
        print("failed.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()