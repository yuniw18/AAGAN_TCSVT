import numpy
import math
import cv2
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
from torch.autograd import Variable
import pytorch_ssim
#import util.
mse_loss = nn.MSELoss()
transform = transforms.Compose([transforms.ToTensor()])

psim = pytorch_ssim.SSIM()
psim = psim.cuda()

def psnr2(img1,img2):
    image1 = Variable(transform(Image.open(img1).convert('RGB')),requires_grad=False)
    image2 = Variable(transform(Image.open(img2).convert('RGB')),requires_grad=False)

    if mse_loss(image1,image2).item() == 0:
        return 0.001
    else:
        return 10 * math.log10(1.0/mse_loss(image1,image2).item())

def SSIM(img1,img2):
    image1 = torch.unsqueeze(transform(Image.open(img1).convert('RGB')),0)
    image2 = torch.unsqueeze(transform(Image.open(img2).convert('RGB')),0)
    image1 = image1.cuda()
    image2 = image2.cuda()
    return psim(image1,image2)

def caculate(image1_path,image2_path):
#    model_num = 5000
    im1_psnr=[]
    im1_ssim=[]
    image_paths = list(os.listdir(image2_path))
    index = 0
    print(len(image_paths))
    for item in image_paths:

#        gen_path = os.path.join(image1_path,str(i) + '_crop.jpg')
 #       target_path = os.path.join(image2_path,str(i) + '_img.jpg')
        print(index) 
        gen_path = os.path.join(image1_path,item)
        target_path = os.path.join(image2_path,item)
#        print(SSIM(gen_path,target_path).item())
#       
        im1_psnr.append(psnr2(gen_path,target_path))
        im1_ssim.append(SSIM(gen_path,target_path).item())
        index = index + 1
    print('psnr:' + str(float(float(sum(im1_psnr))/float(len(im1_psnr))))) 
    print('ssim:' + str(float(sum(im1_ssim)/len(im1_ssim))))     

#caculate('/home/lab-com/edge_new/checkpoints_new/CELEB_TEST/rasgan_aver_hinge_test/results/','/home/lab-com/CelebA/test')
#caculate('/home/lab-com/edge_new/checkpoints_new/PLACES2_TEST/proposed_final/results/','/home/lab-com/places_val')
caculate('/home/lab-com/github/AAGAN/checkpoints/Sample/results_sup/','/home/lab-com/github/CelebA/test')





