import os
from os import listdir
from os.path import join
import cv2
import random
import numpy as np
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def bgr2y(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)
    y, _, _ = cv2.split(ycrcb)
    return y
def bgr2ycb(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)
    y, c, b = cv2.split(ycrcb)
    return y,c,b

CROP_SIZE = 32

class DatasetFromFolder(Dataset):

    def __init__(self):
        
        super(DatasetFromFolder, self).__init__()
        
        lrx4_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_LR_bicubic/x4"
        lrx2_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_LR_bicubic/x2"
        hr_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_HR"

        self.lrx4_list = []
        self.lrx2_list = []
        self.hr_list = []

        lrx4_paths = os.listdir(lrx4_main)
        lrx2_paths = os.listdir(lrx2_main)
        hr_paths = os.listdir(hr_main)

        lrx4_paths.sort()
        lrx2_paths.sort()
        hr_paths.sort()

        for path in hr_paths[1:851]:
            lrx4_path = os.path.join(lrx4_main, path)
            lrx2_path = os.path.join(lrx2_main, path)
            hr_path = os.path.join(hr_main, path)
            self.lrx4_list.append(lrx4_path)
            self.lrx2_list.append(lrx2_path)
            self.hr_list.append(hr_path)
            
    def __getitem__(self, ind):
        
        ip = CROP_SIZE

        lrx4 = cv2.imread(self.lrx4_list[ind].replace(".","x4."))
        lrx2 = cv2.imread(self.lrx2_list[ind].replace(".","x2."))
        hr = cv2.imread(self.hr_list[ind])


        lrx4 = bgr2y(lrx4)
        lrx2 = bgr2y(lrx2)
        hr = bgr2y(hr)

        ih_x4, iw_x4 = lrx4.shape[:2]

        ix_x4 = random.randrange(0, iw_x4 - ip + 1)
        iy_x4 = random.randrange(0, ih_x4 - ip + 1)

        ix_x2 = 2 * ix_x4
        iy_x2 = 2 * iy_x4
        ix_x1 = 4 * ix_x4
        iy_x1 = 4 * iy_x4

        lrx4 = lrx4[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
        lrx2 = lrx2[iy_x2:iy_x2+ip*2,ix_x2:ix_x2+ip*2]
        hr = hr[iy_x1:iy_x1+ip*4,ix_x1:ix_x1+ip*4]

        lrx4 = np.expand_dims(lrx4, axis =0)
        lrx2 = np.expand_dims(lrx2, axis =0)
        hr = np.expand_dims(hr, axis =0)

        lrx4 = torch.from_numpy(lrx4).float()
        lrx2 = torch.from_numpy(lrx2).float()
        hr = torch.from_numpy(hr).float()

        return lrx4, lrx2, hr

    def __len__(self):
            return len(self.lrx4_list)

class DatasetFromFolder2(Dataset):

    def __init__(self):
        
        super(DatasetFromFolder2, self).__init__()
        
        lrx4_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_LR_bicubic/x4"
        lrx2_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_LR_bicubic/x2"
        hr_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_HR"

        self.lrx4_list = []
        self.lrx2_list = []
        self.hr_list = []

        lrx4_paths = os.listdir(lrx4_main)
        lrx2_paths = os.listdir(lrx2_main)
        hr_paths = os.listdir(hr_main)

        lrx4_paths.sort()
        lrx2_paths.sort()
        hr_paths.sort()

        for path in hr_paths[851:861]:
            lrx4_path = os.path.join(lrx4_main, path)
            lrx2_path = os.path.join(lrx2_main, path)
            hr_path = os.path.join(hr_main, path)
            self.lrx4_list.append(lrx4_path)
            self.lrx2_list.append(lrx2_path)
            self.hr_list.append(hr_path)
            
    def __getitem__(self, ind):
        
        ip = 128

        lrx4 = cv2.imread(self.lrx4_list[ind].replace(".","x4."))
        lrx2 = cv2.imread(self.lrx2_list[ind].replace(".","x2."))
        hr = cv2.imread(self.hr_list[ind])


        lrx4 = bgr2y(lrx4)
        lrx2 = bgr2y(lrx2)
        hr = bgr2y(hr)

        
        ih_x4, iw_x4 = lrx4.shape[:2]

        ix_x4 = 10
        iy_x4 = 10
        
        ix_x2 = 2 * ix_x4
        iy_x2 = 2 * iy_x4
        ix_x1 = 4 * ix_x4
        iy_x1 = 4 * iy_x4

        lrx4 = lrx4[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
        lrx2 = lrx2[iy_x2:iy_x2+ip*2,ix_x2:ix_x2+ip*2]
        hr = hr[iy_x1:iy_x1+ip*4,ix_x1:ix_x1+ip*4]
        

        lrx4 = np.expand_dims(lrx4, axis =0)
        lrx2 = np.expand_dims(lrx2, axis =0)
        hr = np.expand_dims(hr, axis =0)

        lrx4 = torch.from_numpy(lrx4).float()
        lrx2 = torch.from_numpy(lrx2).float()
        hr = torch.from_numpy(hr).float()

        return lrx4, lrx2, hr

    def __len__(self):
            return len(self.lrx4_list)
class DatasetFromFolder_plot(Dataset):

    def __init__(self):
        
        super(DatasetFromFolder_plot, self).__init__()
        
        lrx4_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_LR_bicubic/x4"
        lrx2_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_LR_bicubic/x2"
        hr_main = "C:/Users/jhpar/intern/new/DIV2K/DIV2K_train_HR"

        self.lrx4_list = []
        self.lrx2_list = []
        self.hr_list = []

        lrx4_paths = os.listdir(lrx4_main)
        lrx2_paths = os.listdir(lrx2_main)
        hr_paths = os.listdir(hr_main)

        lrx4_paths.sort()
        lrx2_paths.sort()
        hr_paths.sort()

        for path in hr_paths[851:861]:
            lrx4_path = os.path.join(lrx4_main, path)
            lrx2_path = os.path.join(lrx2_main, path)
            hr_path = os.path.join(hr_main, path)
            self.lrx4_list.append(lrx4_path)
            self.lrx2_list.append(lrx2_path)
            self.hr_list.append(hr_path)
            
    def __getitem__(self, ind):
        
        ip = 256

        lrx4 = cv2.imread(self.lrx4_list[ind].replace(".","x4."))
        lrx2 = cv2.imread(self.lrx2_list[ind].replace(".","x2."))
        hr = cv2.imread(self.hr_list[ind])


        lrx4,lrx4_c,lrx4_b = bgr2ycb(lrx4)
        lrx2,lrx2_c,lrx2_b = bgr2ycb(lrx2)
        hr,hr_c,hr_b = bgr2ycb(hr)

        
        ih_x4, iw_x4 = lrx4.shape[:2]

        ix_x4 = 10
        iy_x4 = 10
        
        ix_x2 = 2 * ix_x4
        iy_x2 = 2 * iy_x4
        ix_x1 = 4 * ix_x4
        iy_x1 = 4 * iy_x4

        lrx4 = lrx4[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
        lrx4_c = lrx4_c[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
        lrx4_b = lrx4_b[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
        lrx2 = lrx2[iy_x2:iy_x2+ip*2,ix_x2:ix_x2+ip*2]
        lrx2_c = lrx2_c[iy_x2:iy_x2+ip*2,ix_x2:ix_x2+ip*2]
        lrx2_b = lrx2_b[iy_x2:iy_x2+ip*2,ix_x2:ix_x2+ip*2]
        hr = hr[iy_x1:iy_x1+ip*4,ix_x1:ix_x1+ip*4]
        hr_b = hr_b[iy_x1:iy_x1+ip*4,ix_x1:ix_x1+ip*4]
        hr_c = hr_c[iy_x1:iy_x1+ip*4,ix_x1:ix_x1+ip*4]
        

        lrx4 = np.expand_dims(lrx4, axis =0)
        lrx2 = np.expand_dims(lrx2, axis =0)
        hr = np.expand_dims(hr, axis =0)
        lrx4_c = np.expand_dims(lrx4_c, axis =0)
        lrx2_c = np.expand_dims(lrx2_c, axis =0)
        hr_c = np.expand_dims(hr_c, axis =0)
        lrx4_b = np.expand_dims(lrx4_b, axis =0)
        lrx2_b = np.expand_dims(lrx2_b, axis =0)
        hr_b = np.expand_dims(hr_b, axis =0)
        
        lrx4 = torch.from_numpy(lrx4).float()
        lrx2 = torch.from_numpy(lrx2).float()
        hr = torch.from_numpy(hr).float()
        lrx4_c = torch.from_numpy(lrx4_c).float()
        lrx2_c = torch.from_numpy(lrx2_c).float()
        hr_c = torch.from_numpy(hr_c).float()
        lrx4_b = torch.from_numpy(lrx4_b).float()
        lrx2_b = torch.from_numpy(lrx2_b).float()
        hr_b = torch.from_numpy(hr_b).float()        

        return lrx4, lrx2, hr,lrx4_c, lrx2_c, hr_c,lrx4_b, lrx2_b, hr_b

    def __len__(self):
            return len(self.lrx4_list)
        
class DatasetFromFolder_Set5(Dataset):

    def __init__(self):
        
        super(DatasetFromFolder_Set5, self).__init__()
        
        hr_main = "C:/Users/jhpar/intern/benchmark/Set14/HR"
        lrx4_main = "C:/Users/jhpar/intern/benchmark/Set14/LR_bicubic/X2"
        self.lrx4_list = []
        self.lrx2_list = []
        self.hr_list = []

        lrx4_paths = os.listdir(lrx4_main)
        hr_paths = os.listdir(hr_main)

        lrx4_paths.sort()
        hr_paths.sort()

        for path in hr_paths[0:]:
            hr_path = os.path.join(hr_main, path)
            self.hr_list.append(hr_path)
        for path in lrx4_paths[0:]:
            lrx4_path = os.path.join(lrx4_main, path)
            self.lrx4_list.append(lrx4_path)
                
    def __getitem__(self, ind):
        

        lrx4 = cv2.imread(self.lrx4_list[ind])
        hr = cv2.imread(self.hr_list[ind])
        lrx4,lrx4_c,lrx4_b = bgr2ycb(lrx4)
        hr,hr_c,hr_b = bgr2ycb(hr)

        
        ih_x4, iw_x4 = lrx4.shape[:2]

#         ix_x4 = 10
#         iy_x4 = 10
        
#         ix_x2 = 2 * ix_x4
#         iy_x2 = 2 * iy_x4
#         ix_x1 = 4 * ix_x4
#         iy_x1 = 4 * iy_x4

#         lrx4 = lrx4[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
#         lrx4_c = lrx4_c[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
#         lrx4_b = lrx4_b[iy_x4:iy_x4+ip,ix_x4:ix_x4+ip]
        # if len(hr.shape) == 2:
        #     hr = np.expand_dims(hr,-1)
        #     hr = np.concatenate([hr,hr,hr],-1)
        hr = hr[:ih_x4*2,:iw_x4*2]
        hr_b = hr_b[:ih_x4*2,:iw_x4*2]
        hr_c = hr_c[:ih_x4*2,:iw_x4*2]
        

        lrx4 = np.expand_dims(lrx4, axis =0)
        hr = np.expand_dims(hr, axis =0)
        lrx4_c = np.expand_dims(lrx4_c, axis =0)
        hr_c = np.expand_dims(hr_c, axis =0)
        lrx4_b = np.expand_dims(lrx4_b, axis =0)
        hr_b = np.expand_dims(hr_b, axis =0)
        
        lrx4 = torch.from_numpy(lrx4).float()
        hr = torch.from_numpy(hr).float()
        lrx4_c = torch.from_numpy(lrx4_c).float()
        hr_c = torch.from_numpy(hr_c).float()
        lrx4_b = torch.from_numpy(lrx4_b).float()
        hr_b = torch.from_numpy(hr_b).float()        

        return lrx4, hr,lrx4_c, hr_c,lrx4_b, hr_b

    def __len__(self):
            return len(self.lrx4_list)
