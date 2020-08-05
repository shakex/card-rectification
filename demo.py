from load_model import load_model
from rectify import inference

"""
1. load_model()
根据模型类型，导入存储在硬盘中的模型文件至内存。

Parameters: 
None

Returns:
- model: {UNetRNN}
模型对象，包括模型各层结构和预训练的参数。
- device: {device}
torch.device类对象，表示分配给torch.Tensor进行运算的设备。包含设备类型（"cpu"或"cuda"）和设备序号。

Example:
from load_model import load_model
model, device = load_model()


2. inferecne(input_path, output_path, model, device)
校正推理，对单张图像进行校正处理。

Parameters:
- input_path: {str}
待校正图像路径

- output_path: {str}
图像保存路径

- model: {UNetRNN}
模型对象，包括模型各层结构和预训练的参数。

- device: {device}
torch.device类对象，表示分配给torch.Tensor进行运算的设备。包含设备类型（"cpu"或"cuda"）和设备序号。

Example:
from rectify import inference
from load_model import load_model
input = 'example/card.jpg'
output = 'result/card.png'
model, device = load_model()
inference(input, output, trained_model, device)

"""

if __name__ == "__main__":
    """
    Demo
    """
    input1 = 'example/card1.jpg'
    input2 = 'example/card2.jpg'
    input3 = 'example/card3.jpg'
    input4 = 'example/card4.jpg'
    input5 = 'example/card5.jpg'
    output1 = 'result/card1.png'
    output2 = 'result/card2.png'
    output3 = 'result/card3.png'
    output4 = 'result/card4.png'
    output5 = 'result/card5.png'

    trained_model, device = load_model()
    inference(input1, output1, trained_model, device)
    inference(input2, output2, trained_model, device)
    inference(input3, output3, trained_model, device)
    inference(input4, output4, trained_model, device)
    inference(input5, output5, trained_model, device)

    print("Done.")
