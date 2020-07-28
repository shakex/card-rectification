"""
software: id-card rectification
version: 2.0
"""

import os
import sys
import cv2
import torch
import imutils
import numpy as np
from os.path import join as pjoin
from skimage import io, exposure, img_as_ubyte
from collections import OrderedDict
from imutils.perspective import four_point_transform
from itertools import combinations
from models import get_model
from torchvision import transforms

"""
Parameters Settings
"""
debug = False
debug_dir = 'debug/'
PROCESS_SIZE = 1000
MODEL_INPUT_SIZE = 1000

model_arch = 'UNetRNN'
model_path = "CRDN1000.pkl"


"""
Func1: Image Preprocess
"""

"""
Func2: Edge Detection

Two methods are implemented to detect the edge of input ID-card image:
- Method1: Threshold and Canny based image processing method.
- Method2: An edge detection deep neural network based on conv-rnn and edge-consist loss. (best run with nvidia gpu)
"""


def detect_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    mean_gray = cv2.mean(gray)
    TH_LIGHT = 150
    if mean_gray[0] > TH_LIGHT:
        gray = exposure.adjust_gamma(gray, gamma=6)
        gray = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.02)
        gray = img_as_ubyte(gray)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.medianBlur(closing, 5)
    blurred = cv2.bilateralFilter(blurred, d=0, sigmaColor=15, sigmaSpace=10)

    edged = cv2.Canny(blurred, 75, 200)

    if debug:
        cv2.imwrite(pjoin(debug_dir, name + "_Cannyedge.png"), edged)

    return edged


def load_model():
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    try:
        model = get_model({'arch': model_arch}, n_classes=2).to(device)
        state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)
    except:
        print("Model Error: Model \'" + model_arch + "\' import failed, please check the model file.")
        sys.exit()

    return model, device


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        item_name = k[7:]  # remove `module.`
        new_state_dict[item_name] = v
    return new_state_dict


def get_card_colormap():
    return np.asarray([[0, 0, 0], [255, 255, 255]])


def decode_map(label_mask):
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


def detect_edge_cnn(img, model, device):
    image = cv2.resize(img, (MODEL_INPUT_SIZE, int(MODEL_INPUT_SIZE * img.shape[0] / img.shape[1])),
                       interpolation=cv2.INTER_LINEAR)
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.196, 0.179, 0.323], [0.257, 0.257, 0.401])
        ]
    )
    image = tf(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        img_val = image.to(device)
        res = model(img_val)
        pred = np.squeeze(res.data.max(1)[1].cpu().numpy())
        edged = decode_map(pred)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
        edged = cv2.resize(edged, (PROCESS_SIZE, int(PROCESS_SIZE * img.shape[0] / img.shape[1])),
                           interpolation=cv2.INTER_NEAREST)
        if debug:
            cv2.imwrite(pjoin(debug_dir, name + "_CNNedge.png"), edged)

    return edged


"""
Func3: Four Corner Detection
"""


# get cross point of two lines
def cross_point(line1, line2):
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
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
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


# get angle of two lines
def get_angle(sta_point, mid_point, end_point):
    ma_x = sta_point[0][0] - mid_point[0][0]
    ma_y = sta_point[0][1] - mid_point[0][1]
    mb_x = end_point[0][0] - mid_point[0][0]
    mb_y = end_point[0][1] - mid_point[0][1]
    ab_x = sta_point[0][0] - end_point[0][0]
    ab_y = sta_point[0][1] - end_point[0][1]
    ab_val2 = ab_x * ab_x + ab_y * ab_y
    ma_val2 = ma_x * ma_x + ma_y * ma_y
    mb_val2 = mb_x * mb_x + mb_y * mb_y
    cos_M = (ma_val2 + mb_val2 - ab_val2) / (2 * np.sqrt(ma_val2) * np.sqrt(mb_val2))
    angleAMB = np.arccos(cos_M) / np.pi * 180
    return angleAMB


def checked_valid_transform(approx):
    hull = cv2.convexHull(approx)
    TH_ANGLE = 45
    if len(hull) == 4:
        for i in range(4):
            p1 = hull[(i - 1) % 4]
            p2 = hull[i]
            p3 = hull[(i + 1) % 4]
            angel = get_angle(p1, p2, p3)
            if 90 - TH_ANGLE < angel < 90 + TH_ANGLE:
                continue
            else:
                if debug:
                    print("Detection Error: The detected corners could not form a valid quadrilateral for transformation.")
                raise Exception("Corner points invalid.")
    else:
        if debug:
            print("Detection Error: Could not find four corners from the detected edge.")
        raise Exception("Corner points less than 4.")

    return True


# 获取身份证四个角点
def get_cnt(edged, img, ratio):
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    mask = np.zeros((edged.shape[0], edged.shape[1]), np.uint8)
    mask[10:edged.shape[0] - 10, 10:edged.shape[1] - 10] = 1
    edged = edged * mask

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)

    edgelines = np.zeros(edged.shape, np.uint8)
    cNum = 4

    for i in range(min(cNum, len(cnts))):
        TH = 1 / 20.0
        if cv2.contourArea(cnts[i]) < TH * img.shape[0] * img.shape[1]:
            cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)
        else:
            cv2.drawContours(edgelines, [cnts[i]], 0, (1, 1, 1), -1)
            edgelines = edgelines * edged
            break
        cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)

    if debug:
        cv2.imwrite(pjoin(debug_dir, name + "_edgelines.png"), edgelines)

    lines = cv2.HoughLines(edgelines, 1, np.pi / 180, 200)

    if lines is None or len(lines) < 4:
        if debug:
            print("Detection Error: Could not find enough lines (must more than 4) from the detected edge.")
        raise Exception("Lines not found.")

    if debug:
        lines_draw = np.zeros((len(lines), 4), dtype=int)
        img_draw = img.copy()
        for i in range(0, len(lines)):
            rho, theta = lines[i][0][0], lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            lines_draw[i][0] = int(x0 + 1000 * (-b))
            lines_draw[i][1] = int(y0 + 1000 * (a))
            lines_draw[i][2] = int(x0 - 1000 * (-b))
            lines_draw[i][3] = int(y0 - 1000 * (a))
            cv2.line(img_draw, (lines_draw[i][0], lines_draw[i][1]), (lines_draw[i][2], lines_draw[i][3]), (0, 255, 0),
                     1)
        cv2.imwrite(pjoin(debug_dir, name + '_hough1.png'), img_draw)

    strong_lines = np.zeros([4, 1, 2])
    n2 = 0

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
                if not any(closeness) and n2 < 4 and theta != 0:
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

        if debug:
            cv2.line(img, (lines1[i][0], lines1[i][1]), (lines1[i][2], lines1[i][3]), (0, 255, 0), 3)

    approx = np.zeros((len(strong_lines), 1, 2), dtype=int)
    index = 0
    combs = list((combinations(lines1, 2)))
    for twoLines in combs:
        x1, y1, x2, y2 = twoLines[0]
        x3, y3, x4, y4 = twoLines[1]
        [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
        if 0 < x < img.shape[1] and 0 < y < img.shape[0] and index < 4:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 3)
            approx[index] = (int(x), int(y))
            index = index + 1

    if debug:
        cv2.imwrite(pjoin(debug_dir, name + '_hough2.png'), img)

    if checked_valid_transform(approx):
        return approx * ratio


"""
Func4: Image Postprocess
"""


# 裁剪身份证圆角
def set_corner(img, r):
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


# 结果微调
def finetune(img, ratio):
    offset = int(2 * ratio)
    img = img[offset + 15:img.shape[0] - offset,
          int(offset * 2):img.shape[1] - int(offset * 2), :]
    if img.shape[0] < img.shape[1]:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 856 * 540)))
        r = int(img.shape[1] / 856 * 31.8)
    else:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 540 * 856)))
        r = int(img.shape[1] / 540 * 31.8)
    img = set_corner(img, r)
    return img


"""
Func Main
"""


def inference():
    image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1] not in image_format:
            print("Parameters Error: invalid input or output.")
            sys.exit()
    else:
        print("Parameters Error: invalid input or output.")
        sys.exit()

    global name
    name = os.path.splitext(os.path.basename(input_path))[0]

    trained_model, device = load_model()
    image = cv2.imread(input_path)
    img = cv2.resize(image, (PROCESS_SIZE, int(PROCESS_SIZE * image.shape[0] / image.shape[1])))
    ratio = image.shape[1] / PROCESS_SIZE
    try:
        if debug:
            print("Edge Detection: try method1...")
        edged = detect_edge_cnn(image, trained_model, device)
        corners = get_cnt(edged, img, ratio)
    except:
        try:
            if debug:
                print("Edge Detection: try method2...")
            edged = detect_edge(img)
            corners = get_cnt(edged, img, ratio)
        except:
            print("Failed. {} could not be rectified :(".format(os.path.basename(input_path)))
            sys.exit()

    result = four_point_transform(image, corners.reshape(4, 2))
    result = finetune(result, ratio)
    cv2.imwrite(output_path, result)
    print("Success! Output saved in " + os.path.abspath(output_path))


def inference_all():
    count = 0
    image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    file_list = os.listdir(input_dir)
    file_list.sort()
    for i in range(0, len(file_list)):
        print("[{}/{}] ".format(i+1, len(file_list)), end='')
        in_path = os.path.join(input_dir, file_list[i])

        if os.path.isfile(in_path):
            if os.path.splitext(in_path)[1] not in image_format:
                print("{} is not an acceptable image file, please use .jpg/.jpeg/.bmp/.png as input.".format(file_list[i]))
                continue
        else:
            continue

        global name
        name = os.path.splitext(file_list[i])[0]
        out_path = os.path.join(output_dir, name + ".png")

        trained_model, device = load_model()
        try:
            image = cv2.imread(in_path)
            img = cv2.resize(image, (PROCESS_SIZE, int(PROCESS_SIZE * image.shape[0] / image.shape[1])))
            ratio = image.shape[1] / PROCESS_SIZE
        except:
            continue

        try:
            if debug:
                print("Edge Detection: try method1...")
            edged = detect_edge_cnn(image, trained_model, device)
            corners = get_cnt(edged, img, ratio)
        except:
            try:
                if debug:
                    print("Edge Detection: try method2...")
                edged = detect_edge(img)
                corners = get_cnt(edged, img, ratio)
            except:
                print("Failed. {} could not be rectified :(".format(file_list[i]))
                continue

        result = four_point_transform(image, corners.reshape(4, 2))
        result = finetune(result, ratio)
        cv2.imwrite(out_path, result)
        print("Success! Output saved in " + os.path.abspath(out_path))
        count = count + 1

    print("Done! {}/{} success.".format(count, len(file_list)))


if __name__ == "__main__":
    rectify, input_, output_ = sys.argv
    # input_ = 'example/'
    # output_ = 'result/'
    name = ''

    if os.path.isfile(os.path.abspath(input_)):
        input_path = input_
        output_path = output_
        inference()
    elif os.path.isdir(input_) and os.path.isdir(output_):
        input_dir = input_
        output_dir = output_
        inference_all()
    else:
        print("Parameters Error: invalid input or output.")
