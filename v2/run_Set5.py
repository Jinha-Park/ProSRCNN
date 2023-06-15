import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image

from data import DatasetFromFolder, DatasetFromFolder_Set5
from model import SRCNN
import math
import cv2
import numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Parameters
BATCH_SIZE = 4
NUM_WORKERS = 0 # on Windows, set this variable to 0

trainset = DatasetFromFolder()
testset = DatasetFromFolder_Set5()

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

#model = SRCNN().to(device)
#"model_100.pth"
model = torch.load("model_199.pth").to(device)
avg_psnr = 0
for ind, batch in enumerate(testloader):
#            input, target = batch[0].to(device), batch[1].to(device)
    input,input_c,input_b, target,target_c,target_b = batch[0].to(device), batch[2].to(device),batch[4].to(device),batch[1].to(device),batch[3].to(device),batch[5].to(device)
    
    input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
    input_c = F.interpolate(input_c, scale_factor=2, mode='bilinear', align_corners=False)
    input_b = F.interpolate(input_b, scale_factor=2, mode='bilinear', align_corners=False)

    out2 = model(input)
    out2 = model(out2)
    out2 = model(out2)
    #out2 = model(out2)
    #out2 = model(out2)

    ###
    #print(out2.shape)
    #print(target.shape)
    diff = (out2 - target) / 255
    mse = diff.pow(2).mean()
    psnr = -10 * math.log10(mse.item())
    avg_psnr += psnr
    
    ### PSNR
    out2 = out2.cpu()
    out_img_y = out2[0].detach().numpy()
    input_c = input_c[0].cpu().detach().numpy()
    input_b = input_b[0].cpu().detach().numpy()
    input = input[0].cpu().detach().numpy()
    #out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    

    out_img = cv2.merge([np.uint8(out_img_y[0]), np.uint8(input_c[0]), np.uint8(input_b[0])])
    out_img = cv2.cvtColor(out_img, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite("results/zoomed_output_%d.png"%(ind), out_img)
    out_img = cv2.merge([np.uint8(input[0]), np.uint8(input_c[0]), np.uint8(input_b[0])])
    out_img = cv2.cvtColor(out_img, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite("results/zoomed_input_%d.png"%(ind), out_img)
    
    ind+=1
    
print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")

# Save model
