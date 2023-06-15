import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image

from data import DatasetFromFolder, DatasetFromFolder2
from model import SRCNN
import math

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Parameters
BATCH_SIZE = 4
NUM_WORKERS = 0 # on Windows, set this variable to 0

trainset = DatasetFromFolder()
testset = DatasetFromFolder2()

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model = torch.load("model_99.pth").to(device)
criterion = nn.L1Loss()
criterion_MSE = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) ## 1e-4, 1e-3, 1e-2, 1e-1?? 

for epoch in range(args.nb_epochs):

    # Train
    epoch_loss = 0
    for iteration, batch in enumerate(trainloader):
        lrx4, lrx2, hr = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        lrx4 = F.interpolate(lrx4, scale_factor=4, mode='bilinear', align_corners=False)
        lrx2 = F.interpolate(lrx2, scale_factor=2, mode='bilinear', align_corners=False)
        optimizer.zero_grad()

        outx2 = model(lrx4)
        outx1 = model(outx2)
        loss = criterion(outx2, lrx2) + criterion(outx1, hr)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")

    # Test
    avg_psnr = 0
    with torch.no_grad():
        ind = 0
        for ind, batch in enumerate(testloader):
#            input, target = batch[0].to(device), batch[1].to(device)
            input, target = batch[0].to(device), batch[2].to(device)
            input = F.interpolate(input, scale_factor=4, mode='bilinear', align_corners=False)

            out2 = model(input)
            
            ###
            diff = (out2 - target) / 255
            mse = diff.pow(2).mean()
            psnr = -10 * math.log10(mse.item())
            avg_psnr += psnr
            ind+= 1
            ### 
            input = input[0] #torch.Size([3,28,28]
            save_image(input/255, 'results/input_%d.png'%(ind))
            out2 = out2[0] #torch.Size([3,28,28]
            save_image(out2/255, 'results/out2_%d.png'%(ind))
            target = target[0] #torch.Size([3,28,28]
            save_image(target/255, 'results/target_%d.png'%(ind))
            
            
    print(f"Average PSNR: {avg_psnr / ind} dB.")

    # Save model
    torch.save(model, f"model_{epoch + 100}.pth")
