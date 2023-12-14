import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image
import numpy as np
import cv2

path = "both.jpg"     
raw_img = cv2.imread("./data/"+path)
preprocess = transforms.ToTensor()
img = preprocess(raw_img)
img = torch.unsqueeze(img, 0)
(_,_,W,H)=img.size()

model = torch.load('torch_alex.pth')

grad_block = []	# 存放grad图
feaure_block = []	# 存放特征图

# 获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 获取特征层的函数
def farward_hook(module, input, output):
    feaure_block.append(output)

# 已知原图、梯度、特征图，开始计算可视化图
def cam_show_img(img, feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加
    grads = grads.reshape([grads.shape[0], -1])
    # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
    weights = np.mean(grads, axis=1)	
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]	# 特征图加权和
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    # cam.dim=2 heatmap.dim=3
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)	# 伪彩色

    # print(heatmap.shape)
    # print(img.size)

    cam_img = 0.3 * heatmap + 0.7 * img

    cv2.imwrite("./out/"+path, cam_img)


# layer_name=model.features[18][1]
model.features.register_forward_hook(farward_hook)
model.features.register_full_backward_hook(backward_hook)

# forward 
# 在前向推理时，会生成特征图和预测值
output = model(img)
max_idx = np.argmax(output.data.numpy())

# backward
model.zero_grad()
# 取最大类别的值作为loss，这样计算的结果是模型对该类最感兴趣的cam图
class_loss = output[0, max_idx]	
class_loss.backward()	# 反向梯度，得到梯度图

# grads
grads_val = grad_block[0].data.numpy().squeeze()
fmap = feaure_block[0].data.numpy().squeeze()

# save cam
cam_show_img(raw_img, fmap, grads_val)
