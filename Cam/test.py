# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models
import argparse
import cv2
import torchvision.transforms as transforms

from cam.layercam import *
from utils import *

def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of LayerCAM')
    parser.add_argument("--img_path", type=str, default='both.jpg', help='Path of test image')
    parser.add_argument("--layer_id", type=list, default=[0,3,6,9,12], help='The cam generation layer')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    path = args.img_path     
    raw_img = cv2.imread("./images/"+path)
    preprocess = transforms.ToTensor()
    img = preprocess(raw_img)
    img = torch.unsqueeze(img, 0)

    model = torch.load('torch_alex.pth')
    
    for i in range(len(args.layer_id)):
        layer_name = 'features_' + str(args.layer_id[i])
        model_dict = dict(type='vgg16', arch=model, layer_name=layer_name, input_size=(224, 224))
        layercam = LayerCAM(model_dict)
        predicted_class = model(img).max(1)[-1]

        layercam_map = layercam(img)
        basic_visualize(img.detach(), layercam_map.type(torch.FloatTensor),save_path='./vis/'+path+'_stage_{}.png'.format(i+1))
