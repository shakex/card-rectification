# 身份证图片校正程序 (beta)
主要功能：针对输入的身份证件图像进行透视变化校正，输出原分辨率大小的校正图像。

## Features
1. 针对和身份证卡片颜色相近的背景可以更精确的检测;
2. 再次提升了身份证边缘识别的精度，裁剪出的照片文字倾斜的现象（校正平整度）有了改善。

## Requirements
- python==3.5.2
- opencv-python==3.4.0.12
- scikit-image==0.13.0
- numpy==1.12.1
- imutils==0.5.3

## Usage
1. `$ pip install -r requirements.txt`
2. 进入 `/card-rectify` 目录
3. `$ python recitify.py [input_path] [output_path]`
    - `input_path`: 待校正图像的位置
    - `output_path`: 校正后图像的保存位置
    - e.g. `$ python recitify.py example/card1.jpg result/card1_res.jpg`

## Unsolved Situation
1. 背景较为复杂的情况；
2. 白色背景（白纸等）或背景与身份证本身比较接近的情况；
3. 身份证拍摄不完整或有边缘遮挡的情况；
4. 拍摄倾角过大的情况；
5. 因光照产生明显边缘阴影的情况；
6. 身份证图像非正立，旋转角度大于45度的情况。






