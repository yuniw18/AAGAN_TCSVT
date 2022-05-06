import sys
import os
import torchvision
import threading
import torch
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *
import math
import torch.nn as nn
import numpy as np
from scipy.misc import imread,imsave
import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
from my_args import args
from discriminator import Discriminator,Discriminator_edge
from MB_test import test
from test_load import *
import pytorch_ssim


mse_loss = nn.MSELoss()

psim = pytorch_ssim.SSIM()
psim = psim.cuda()


def psnr2(img1,img2):
    return 10 * math.log10(1.0/mse_loss(img1,img2).item())

def SSIM(img1,img2):
    image1 = img1.cuda()
    image2 = img2.cuda()
    return psim(image1,image2)


def test():
    torch.manual_seed(args.seed)

    model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=True)
###################################################################################################################################################################
    if (args.MULTI_FRAME == True) and (args.Auto ==True) and (args.USE_SPEC == True):
        discriminator = Discriminator(in_channels=9,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
        print("****************************************************************************************")
        print("Proposed_multi_frame_auto_spec")
    elif (args.MULTI_FRAME == False) and (args.Auto == True) and (args.USE_SPEC == True):
        discriminator = Discriminator(in_channels=3,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
    elif (args.MULTI_FRAME == True) and (args.Auto == False) and (args.USE_SPEC == True):
        discriminator = Discriminator_edge(in_channels=9,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
        print("**************************************************************************************")
        print("Multi - NoAUto - USE_SPEC")
    elif (args.Auto == True) and (args.USE_SPEC == False):
        discriminator = Discriminator(in_channels=3,use_spectral_norm=False,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
    elif (args.Auto == False) and (args.USE_SPEC == True):
        discriminator = Discriminator_edge(in_channels=3,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
    elif (args.Auto == False) and (args.USE_SPEC == False):
        discriminator = Discriminator_edge(in_channels=3,use_spectral_norm=False,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
    else: 
        print("Wrong Discriminator configuration !!")
####################################################################################################################################################################
    discriminator.cuda()

    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()

    if not args.SAVED_MODEL==None:
        SAVED_GEN ='../model_weights/'+ args.uid + "/epoch0" + ".pth"
#        SAVED_GEN ='../model_weights/'+ args.SAVED_MODEL + "/best" + ".pth"
#        SAVED_DIS ='../model_weights/'+ args.SAVED_MODEL + "/epoch_dis0" + ".pth"
#        print("Fine tuning on " +  args.SAVED_MODEL)
        if not  args.use_cuda:
            pretrained_dict = torch.load(SAVED_GEN, map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(SAVED_GEN)
            print("#################################################################################################")
            print(args.SAVED_MODEL)
            print("#################################################################################################")
 

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        pretrained_dict = None
        
    torch.manual_seed(1593665876)
    torch.cuda.manual_seed_all(4099049913103886)    
    
    test_set = TestDataset(args.TESTDATA_PATH)   
    print(args.TESTDATA_PATH)    
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,shuffle=True,
                                             num_workers=args.workers, pin_memory=True if args.use_cuda else False)
    print(len(test_loader))
    # if not args.lr == 0:
    print("train the interpolation net")
    
    if args.Fine_Tuning == False: 
        a = 1 
    
    elif args.Fine_Tuning == True: 
        print("--------------------------------Do Fine Tuning with GAN---------------------------------")

    print("*********Start Training********")
#    print("EPOCH is: "+ str(int(len(train_set) / args.batch_size )))
#    print("Num of EPOCH is: "+ str(args.numEpoch))
    def count_network_parameters(model):

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])

        return N
    print("Num. of model parameters is :" + str(count_network_parameters(model)))
    if hasattr(model,'flownets'):
        print("Num. of flow model parameters is :" +
              str(count_network_parameters(model.flownets)))
    if hasattr(model,'initScaleNets_occlusion'):
        print("Num. of initScaleNets_occlusion model parameters is :" +
              str(count_network_parameters(model.initScaleNets_occlusion) +
                  count_network_parameters(model.initScaleNets_occlusion1) +
        count_network_parameters(model.initScaleNets_occlusion2)))
    if hasattr(model,'initScaleNets_filter'):
        print("Num. of initScaleNets_filter model parameters is :" +
              str(count_network_parameters(model.initScaleNets_filter) +
                  count_network_parameters(model.initScaleNets_filter1) +
        count_network_parameters(model.initScaleNets_filter2)))
    if hasattr(model, 'ctxNet'):
        print("Num. of ctxNet model parameters is :" +
              str(count_network_parameters(model.ctxNet)))
    if hasattr(model, 'depthNet'):
        print("Num. of depthNet model parameters is :" +
              str(count_network_parameters(model.depthNet)))
    if hasattr(model,'rectifyNet'):
        print("Num. of rectifyNet model parameters is :" +
              str(count_network_parameters(model.rectifyNet)))

    
    l1_loss = nn.L1Loss()
 
    
    
    training_losses = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    ikk = 0
    with torch.no_grad():
        print("\t\t**************Start Validation*****************")
        #Turn into evaluation mode

        val_total_losses = AverageMeter()
        val_total_pixel_loss = AverageMeter()
        val_total_PSNR_loss = AverageMeter()
        val_total_tv_loss = AverageMeter()
        val_total_pws_loss = AverageMeter()
        val_total_sym_loss = AverageMeter()
        psnr = []
        ssim = []
        for i, (X0,X1,y) in enumerate(test_loader):
            if i>= 10000:
                break
            t = 0                                   # indicates epoch
            with torch.no_grad():
                X0 = X0.cuda() if args.use_cuda else X0
                X1 = X1.cuda() if args.use_cuda else X1
                y = y.cuda() if args.use_cuda else y

                diffs, offsets,filters,occlusions,cur_output,cur_rectified,ground_truth = model(torch.stack((X0,y,X1),dim = 0))

                pixel_loss, offset_loss,sym_loss = part_loss(diffs, offsets, occlusions, [X0,X1],epsilon=args.epsilon)

                outputs = cur_rectified
                images = ground_truth
                dis_input_real = images
                dis_input_fake = outputs.detach()
 
                if args.MULTI_FRAME:
                    dis_input_real = torch.cat((X0,images,X1),dim=1)
                    dis_input_fake = torch.cat((X0,outputs.detach(),X1),dim=1) 
                    dis_real, _ = discriminator(dis_input_real)                    # in: [rgb(3)]
                    dis_fake, _ = discriminator(dis_input_fake)                    # in: [rgb(3)]
                else:
                    dis_real, _ = discriminator(dis_input_real)                    # in: [rgb(3)]
                    dis_fake, _ = discriminator(dis_input_fake)                    # in: [rgb(3)]



                val_total_loss = sum(x * y for x, y in zip(args.alpha, pixel_loss))

                per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2,
                                                                    dim=1),dim=1),dim=1)
                per_sample_pix_error = per_sample_pix_error.data # extract tensor
                psnr_loss = torch.mean(20 * torch.log(1.0/torch.sqrt(per_sample_pix_error)))/torch.log(torch.Tensor([10]))
                val_total_losses.update(val_total_loss.item(),args.batch_size)
                val_total_pixel_loss.update(pixel_loss[args.save_which].item(), args.batch_size)
                val_total_tv_loss.update(offset_loss[0].item(), args.batch_size)
                val_total_sym_loss.update(sym_loss[0].item(), args.batch_size)
                val_total_PSNR_loss.update(psnr_loss[0],args.batch_size)
                
                sample_path = args.SAMPLE_PATH + '/UCF'+str(t)
                log_path = os.path.join(args.SAMPLE_PATH,'UCF.log')
                if not os.path.exists(sample_path):
                    os.mkdir(sample_path) 

                if True:
                    gt_path = os.path.join(sample_path,'gt_%d.png'%(i))
                    out_path = os.path.join(sample_path,'out_%d.png'%(i))
                    disp_path = os.path.join(sample_path,'disp_%d.png'%(i))
                    disp_gt_path = os.path.join(sample_path,'disp_gt_%d.png'%(i))
                    disp_real_path = os.path.join(sample_path,'disp_real_%d.png'%(i))
                
                    diff = ground_truth - cur_rectified                
                    
                    cur_rectified = cur_rectified[:,:,8:8+240,64:64+320]
                    ground_truth = ground_truth[:,:,8:8+240,64:64+320]

                    torchvision.utils.save_image(cur_rectified, out_path)
                    torchvision.utils.save_image(ground_truth,gt_path)                
                    psnr.append(psnr_loss)
                    ssim.append(SSIM(cur_rectified,ground_truth).item())

                    
                    



        f = open(log_path,'a')
        f.write( 'psnr:' + str(float(float(sum(psnr))/float(len(psnr))))+ "\n")
        f.write( 'ssim:' + str(float(sum(ssim)/len(ssim)))+ "\n")
        f.close()
 

        print("\t\tFinished an epoch, Check and Save the model weights")


    print("*********Finish Training********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=test)
    thread.start()
    thread.join()

    exit(0)
