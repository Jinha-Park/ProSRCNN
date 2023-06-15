import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
from math import log10

def bgr2ycrcb(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    return y, cr, cb

parser = argparse.ArgumentParser(description='SRCNN run parameters')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--path1', type=str, required=True)
parser.add_argument('--path2', type=str, required=True)
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

img_list = os.listdir(args.path1)
tar_list = os.listdir(args.path2)
#bic_list = os.listdir(args.path3)

avg_psnr = 0

for i in range(1,15):
    
    img = cv2.imread("C:/Users/jhpar/intern/new/SRCNN/Set14x4/" + img_list[i])
    tar = cv2.imread("C:/Users/jhpar/intern/new/SRCNN/Set14_cropped/" + tar_list[i])
    #bic = cv2.imread("C:/Users/jhpar/intern/new/SRCNN/Set14_bicubic/" + bic_list[i])
    
    img = cv2.resize(img, dsize = (0,0), fx = args.zoom_factor, fy = args.zoom_factor, interpolation = cv2.INTER_CUBIC)  # first, we upscale the image via bicubic interpolation
    y, cr, cb = bgr2ycrcb(img)
    yt, _, _ = bgr2ycrcb(tar)
    #yb, _, _ = bgr2ycrcb(bic)

    #img_to_tensor = transforms.ToTensor()
    #input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

    y = np.expand_dims(y, axis=0)
    y = np.expand_dims(y, axis=0)
    input = torch.from_numpy(y).float()
    #tar = torch.from_numpy(yt).float()
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
    print(device)
    model = torch.load(args.model).to(device)
    input = input.to(device)

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    #out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    
    #out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    #out_img = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')  # we merge the output of our network with the upscaled Cb and Cr from before
                                                                        # before converting the result in RGB

    out_img = cv2.merge([np.uint8(out_img_y[0]), cr, cb])
    out_img = cv2.cvtColor(out_img, cv2.COLOR_YCrCb2BGR)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

    mse =np.mean((out_img_y[0] - yt) **2)
    #mse =np.mean((yb - yt) **2)
    psnr = 20 * log10(255.0 / math.sqrt(mse))
    print(psnr)
    avg_psnr += psnr
    
    cv2.imwrite(f"results/zoomed_" + img_list[i], out_img)
    
print(f"Average PSNR: {avg_psnr / 14} dB.")