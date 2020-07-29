# 身份证图片校正程序 (v2.0)
主要功能：针对输入的身份证件图像进行透视变化校正，输出原分辨率大小的校正图像。基于Python开发。
- 校正示例：
![example](example.jpg)

## 更新特性
1. 结合了深度学习，针对白色背景、背景复杂、光源变化的情况，可以有效地识别证件边缘，提高校正的准确率；
2. 针对不同长宽的输入图像，最后结果都会统一输出为二代身份证标准格式。


## 依赖包
- pytorch
- torchvision
- opencv-python
- scikit-image
- imutils
- numpy
- imutils


## 使用方法
### 安装（使用CPU）
1. `$ pip install -r requirements.txt`

### 安装（使用GPU）
1. 安装cuda：参考 https://developer.nvidia.com/cuda-downloads
2. `$ pip install -r requirements.txt`

### 运行
1. 进入 `Card-Rectification/` 目录
2. 单张图片处理：`$ python rectify.py [input_path] [output_path]`
    - `input_path`: 待校正图像的位置
    - `output_path`: 校正后图像的保存位置
    - e.g. `$ python rectify.py example/card1.jpg result/card1_res.png`
3. 批处理：`$ python rectify.py [input_dir] [output_dir]
    - `input_dir`: 待校正图像目录
    - `output_dir`: 校正后图像保存目录
    - 默认保存图片为.png格式
    - e.g. `$ python rectify.py example/ result/`
* GPU使用：在GPU可用的情况下，程序会优先使用GPU运行；否则将采用CPU运行。

## 尚未解决的情况
1. 身份证拍摄不完整或有边缘遮挡的情况；
2. 对于身份证正反面识别并对颠倒的校正结果进行180旋转调整。




