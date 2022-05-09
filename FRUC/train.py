import sys
import os
import torchvision
import threading
import torch
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *
from loss import AdversarialLoss, PerceptualLoss, StyleLoss,GDL_loss
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
from discriminator import Discriminator,Discriminator_edge,UNET_Discriminator
from MB_test import test
import math
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


def train():
    torch.manual_seed(args.seed)

    model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=True)
###################################################################################################################################################################
    if (args.MULTI_FRAME == True) and (args.GAN_loss == 'UNET_loss'):
        discriminator = UNET_Discriminator(in_channels=9,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
        print("****************************************************************************************")
        print("UNET_multiframe")
    elif (args.MULTI_FRAME == False) and (args.GAN_loss == 'UNET_loss'):
        discriminator = UNET_Discriminator(in_channels=3,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
    elif (args.MULTI_FRAME == True) and (args.GAN_loss == 'rs_proposed'):
        discriminator = UNET_Discriminator(in_channels=9,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
        print("****************************************************************************************")
        print("UNET_multiframe")
    elif (args.MULTI_FRAME == False) and (args.GAN_loss == 'rs_proposed'):
        discriminator = UNET_Discriminator(in_channels=3,use_spectral_norm=True,use_sigmoid=((args.GAN_loss !='hinge') or (args.GAN_loss !='rasgan_aver_hinge')))
 
    elif (args.MULTI_FRAME == True) and (args.Auto ==True) and (args.USE_SPEC == True):
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
        SAVED_GEN ='../model_weights/'+ args.SAVED_MODEL + "/best" + ".pth"
        print("Fine tuning on " +  args.SAVED_MODEL)
        if not  args.use_cuda:
            pretrained_dict = torch.load(SAVED_GEN, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(SAVED_GEN)
            print("#################################################################################################")
            print(args.SAVED_MODEL)
            print("#################################################################################################")
 
            # model.load_state_dict(torch.load(args.SAVED_MODEL))

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        pretrained_dict = None

    if type(args.datasetName) == list:
        train_sets, test_sets = [],[]
        for ii, jj in zip(args.datasetName, args.datasetPath):
            tr_s, te_s = datasets.__dict__[ii](jj, split = args.dataset_split,single = args.single_output, task = args.task)
            train_sets.append(tr_s)
            test_sets.append(te_s)
        train_set = torch.utils.data.ConcatDataset(train_sets)
        test_set = torch.utils.data.ConcatDataset(test_sets)
    else:
        train_set, test_set = datasets.__dict__[args.datasetName](args.datasetPath)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = args.batch_size,
        sampler=balancedsampler.RandomBalancedSampler(train_set, int(len(train_set) / args.batch_size )),
        num_workers= args.workers, pin_memory=True if args.use_cuda else False)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                             num_workers=args.workers, pin_memory=True if args.use_cuda else False)
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))


    # if not args.lr == 0:
    print("train the interpolation net")
    
    if args.Fine_Tuning == False: 
        optimizer = torch.optim.Adamax([
                {'params': model.initScaleNets_filter.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.initScaleNets_filter1.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.initScaleNets_filter2.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.ctxNet.parameters(), 'lr': args.ctx_lr_coe * args.lr},
                {'params': model.flownets.parameters(), 'lr': args.flow_lr_coe * args.lr},
                {'params': model.depthNet.parameters(), 'lr': args.depth_lr_coe * args.lr},
                {'params': model.rectifyNet.parameters(), 'lr': args.rectify_lr}
            ],
                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    
    
    elif args.Fine_Tuning == True: 
        print("--------------------------------Do Fine Tuning with GAN---------------------------------")
        optimizer = torch.optim.Adamax([
                {'params': model.rectifyNet.parameters(), 'lr': args.rectify_lr}
            ],
                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

        dis_optimizer = torch.optim.Adam(
            params=discriminator.parameters(),
            lr=float(args.rectify_lr) * float(args.D2G_LR),
            betas=(0.9, 0.999)
        )


    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=args.factor, patience=args.patience,verbose=True)

    print("*********Start Training********")
    print("LR is: "+ str(float(optimizer.param_groups[0]['lr'])))
    print("EPOCH is: "+ str(int(len(train_set) / args.batch_size )))
    print("Num of EPOCH is: "+ str(args.numEpoch))
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
    adversarial_loss = AdversarialLoss(type=args.GAN_loss).cuda()
    mse_loss = nn.MSELoss() 
    
    
    training_losses = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    ikk = 0
    for kk in optimizer.param_groups:
        if kk['lr'] > 0:
            ikk = kk
            break

    for t in range(args.numEpoch):
        print("The id of this in-training network is " + str(args.uid))
        print(args)
        #Turn into training mode
        for i, (X0_half,X1_half, y_half) in enumerate(train_loader):
            model = model.train()
            discriminator = discriminator.train()
            
            if i >= int(len(train_set) / args.batch_size ):
                break

            X0_half = X0_half.cuda() if args.use_cuda else X0_half
            X1_half = X1_half.cuda() if args.use_cuda else X1_half
            y_half = y_half.cuda() if args.use_cuda else y_half

            X0 = Variable(X0_half, requires_grad= False)
            X1 = Variable(X1_half, requires_grad= False)
            y  = Variable(y_half,requires_grad= False)

            diffs, offsets,filters,occlusions,cur_output,cur_rectified,ground_truth = model(torch.stack((X0,y,X1),dim = 0))

            pixel_loss, offset_loss, sym_loss = part_loss(diffs,offsets,occlusions, [X0,X1],epsilon=args.epsilon)
         
######################################################################################################################################################
            dis_loss = 0
            gen_loss = 0
            outputs = cur_rectified
            images = ground_truth
            dis_input_real = images
            dis_input_fake = outputs.detach()
            
            if args.MULTI_FRAME:
                if args.GAN_loss == 'UNET_loss':
                    dis_input_real = torch.cat((X0,images,X1),dim=1)
                    dis_input_fake = torch.cat((X0,outputs.detach(),X1),dim=1) 
                    dis_real_dec, dis_real_enc = discriminator(dis_input_real)                    # in: [rgb(3)]
                    dis_fake_dec, dis_fake_enc = discriminator(dis_input_fake)                    # in: [rgb(3)]

               
                else:
                    dis_input_real = torch.cat((X0,images,X1),dim=1)
                    dis_input_fake = torch.cat((X0,outputs.detach(),X1),dim=1) 
                    dis_real, _ = discriminator(dis_input_real)                    # in: [rgb(3)]
                    dis_fake, _ = discriminator(dis_input_fake)                    # in: [rgb(3)]

           
            else:
               dis_real, _ = discriminator(dis_input_real)                    # in: [rgb(3)]
               dis_fake, _ = discriminator(dis_input_fake)                    # in: [rgb(3)]


            if args.GAN_loss == 'proposed':

                zero = torch.zeros_like(dis_real,requires_grad=False)            
                dis_real_loss = torch.mean(torch.abs((dis_real - zero)))

                sel_images = dis_fake * images + (1.0 - dis_fake) * outputs.detach()
                dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
                disparity = torch.abs(images - outputs.detach())
                disp_loss = torch.mean(torch.abs((dis_fake - args.DISP_WEIGHT * disparity))) 
                
                
                dis_loss += (dis_real_loss  
                + args.FAKE_LOSS_WEIGHT * dis_fake_loss  
                + args.DISP_LOSS_WEIGHT * disp_loss)
                
                dis_optimizer.zero_grad()
                dis_loss.backward()
                dis_optimizer.step()
            
            elif args.GAN_loss == 'UNET_loss':
                dis_real_loss_enc = adversarial_loss(dis_real_enc, True, True)
                dis_fake_loss_enc = adversarial_loss(dis_fake_enc, False, True)
                dis_real_loss_dec = adversarial_loss(dis_real_dec, True, True)
                dis_fake_loss_dec = adversarial_loss(dis_fake_dec, False, True)
            
                dis_loss += (dis_real_loss_enc + dis_fake_loss_enc + dis_real_loss_dec + dis_fake_loss_dec) 
 
                dis_optimizer.zero_grad()
                dis_loss.backward()
                dis_optimizer.step()

            elif args.GAN_loss == 'rs_proposed':
            
                zero = torch.zeros_like(dis_real,requires_grad=False)            
                one = torch.ones_like(dis_fake,requires_grad = False)
                
                dis_real = torch.nn.ReLU()(1.0 + (dis_real dis_fake))
                dis_real_loss = torch.mean(torch.abs((dis_real - zero)))
            
                dis_fake_sel = torch.nn.ReLU()(dis_fake - dis_real)
                dis_fake_sel_loss = torch.mean(torch.abs((dis_fake_sel - one))) # Not used

                sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
                dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
                disparity = torch.abs(images - outputs.detach())
                disp_loss = torch.mean(torch.abs((dis_fake - args.DISP_WEIGHT * disparity)))

                dis_loss += (dis_real_loss  
                + args.FAKE_LOSS_WEIGHT * dis_fake_loss 
                + args.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss  # Not used 
                + args.DISP_LOSS_WEIGHT * disp_loss)

                dis_optimizer.zero_grad()
                dis_loss.backward()
                dis_optimizer.step()
   
            elif args.GAN_loss == 'rasgan_aver_lsgan':
                dis_real_loss = torch.mean((dis_real - torch.mean(dis_fake) - 1.0)**2) 
                dis_fake_loss = torch.mean((dis_fake - torch.mean(dis_real) + 1.0)**2)

                dis_loss += dis_fake_loss + dis_real_loss
                
                dis_optimizer.zero_grad()
                dis_loss.backward()
                dis_optimizer.step()
 
            else:
                print("NO GAN LOSS APPLIED")
                break



            if args.GAN_loss == 'proposed':
                if args.MULTI_FRAME:
                    gen_output = torch.cat((X0,cur_rectified,X1),dim=1)
                    gen_fake, _ = discriminator(gen_output)                    # in: [rgb(3)]

                else:
                    gen_output = cur_rectified
                    gen_fake, _ = discriminator(gen_output)                    # in: [rgb(3)]


                gen_gan_loss = torch.mean(torch.abs((gen_fake)))
                gen_loss += gen_gan_loss * args.GAN_LOSS_WEIGHT                                # set Gan loss weight 
            elif args.GAN_loss == 'rs_proposed':
                if args.MULTI_FRAME:
                    gen_output = torch.cat((X0,cur_rectified,X1),dim=1)
                    gen_fake, _ = discriminator(gen_output)                    # in: [rgb(3)]

                else:
                    gen_output = cur_rectified
                    gen_fake, _ = discriminator(gen_output)                    # in: [rgb(3)]


                gen_gan_loss = torch.mean(torch.abs((gen_fake)))
                gen_loss += gen_gan_loss * args.GAN_LOSS_WEIGHT                                # set Gan loss weight 
            elif args.GAN_loss == 'UNET_loss':
                if args.MULTI_FRAME:
                    gen_output = torch.cat((X0,cur_rectified,X1),dim=1)
                    gen_fake_dec, gen_fake_enc = discriminator(gen_output)                    # in: [rgb(3)]
                else:
                    gen_output = cur_rectified
                    gen_fake_dec, gen_fake_enc = discriminator(gen_output)                    # in: [rgb(3)]

                gen_gan_loss_enc = adversarial_loss(gen_fake_enc,True,False) 
                gen_gan_loss_dec = adversarial_loss(gen_fake_dec,True,False)
                gen_gan_loss = ( gen_gan_loss_enc + gen_gan_loss_dec)
    
                gen_loss += gen_gan_loss *  args.GAN_LOSS_WEIGHT                              



            elif args.GAN_loss == 'rasgan_aver_lsgan':
                if args.MULTI_FRAME:
                    gen_output = torch.cat((X0,cur_rectified,X1),dim=1)
                    real_out = torch.cat((X0,images,X1),dim=1)
 
                    gen_fake, _ = discriminator(gen_output)                    # in: [rgb(3)]
                    gen_real, _ = discriminator(real_out) 


                else:
                    gen_output = cur_rectified
                    real_out = images

                    gen_fake, _ = discriminator(gen_output)                    # in: [rgb(3)]
                    gen_real, _ = discriminator(real_out) 


                gen_gan_loss = torch.mean((gen_real - torch.mean(gen_fake) + 1.0)**2) + torch.mean((gen_fake - torch.mean(gen_real) - 1.0) ** 2)
                gen_loss += gen_gan_loss *  args.GAN_LOSS_WEIGHT


                 

# Code backward function (add dis and gen loss with total_loss) & check what total loss means
# set optimizer for discrimiantor and do training

####################################################################################################################################################
 

            total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
            total_loss += gen_loss


            training_losses.update(total_loss.item(), args.batch_size)
            if i % max(1, int(int(len(train_set) / args.batch_size )/500.0)) == 0:

                print("Ep [" + str(t) +"/" + str(i) +
                                    "]\tl.r.: " + str(round(float(ikk['lr']),7))+
                                    "\tPix: " + str([round(x.item(),5) for x in pixel_loss]) +
                                    "\tTV: " + str([round(x.item(),4)  for x in offset_loss]) +
                                    "\tSym: " + str([round(x.item(), 4) for x in sym_loss]) +
                                    "\tTotal: " + str([round(x.item(),5) for x in [total_loss]]) +
                                    "\tAvg. Loss: " + str([round(training_losses.avg, 5)]) +
                                    "\tGen. Loss:" + str(gen_loss.item()))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

       
        torch.save(model.state_dict(), args.save_path + "/epoch" + str(t) +".pth")
        torch.save(discriminator.state_dict(),args.save_path + "/epoch_dis" + str(t) + ".pth")



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
        for i, (X0,X1,y) in enumerate(val_loader):
            if i >=  int(len(test_set)/ args.batch_size):
                break

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
                
                sample_path = args.SAMPLE_PATH + '/Vimeo'+str(t)
                log_path = os.path.join(args.SAMPLE_PATH,'Vimeo.log')
 
                if not os.path.exists(sample_path):
                    os.mkdir(sample_path) 

                if False:
                    gt_path = os.path.join(sample_path,'gt_%d.png'%(i))
                    out_path = os.path.join(sample_path,'out_%d.png'%(i))
                    disp_path = os.path.join(sample_path,'disp_%d.png'%(i))
                    disp_gt_path = os.path.join(sample_path,'disp_gt_%d.png'%(i))
                    disp_real_path = os.path.join(sample_path,'disp_real_%d.png'%(i))
                
                    diff = ground_truth - cur_rectified                

                    torchvision.utils.save_image(cur_rectified, out_path)
                    torchvision.utils.save_image(ground_truth,gt_path)                
                    torchvision.utils.save_image(diff,disp_gt_path)
                    torchvision.utils.save_image(dis_fake,disp_path)
                    torchvision.utils.save_image(dis_real,disp_real_path)
                psnr.append(psnr_loss)
                ssim.append(SSIM(cur_rectified,ground_truth).item())

        f = open(log_path,'a')
        f.write( 'psnr:' + str(float(float(sum(psnr))/float(len(psnr))))+ "\n")
        f.write( 'ssim:' + str(float(sum(ssim)/len(ssim)))+ "\n")
        f.close()
 


        print("\nEpoch " + str(int(t)) +
              "\tlearning rate: " + str(float(ikk['lr'])) +
              "\tAvg Training Loss: " + str(round(training_losses.avg,5)) +
              "\tValidate Loss: " + str([round(float(val_total_losses.avg), 5)]) +
              "\tValidate PSNR: " + str([round(float(val_total_PSNR_loss.avg), 5)]) +
              "\tPixel Loss: " + str([round(float(val_total_pixel_loss.avg), 5)]) +
              "\tTV Loss: " + str([round(float(val_total_tv_loss.avg), 4)]) +
              "\tPWS Loss: " + str([round(float(val_total_pws_loss.avg), 4)]) +
              "\tSym Loss: " + str([round(float(val_total_sym_loss.avg), 4)])
              )

        auxiliary_data.append([t, float(ikk['lr']),
                                   training_losses.avg, val_total_losses.avg, val_total_pixel_loss.avg,
                                   val_total_tv_loss.avg,val_total_pws_loss.avg,val_total_sym_loss.avg])
        
        numpy.savetxt(args.log, numpy.array(auxiliary_data), fmt='%.8f', delimiter=',')
        training_losses.reset()

        print("\t\tFinished an epoch, Check and Save the model weights")
            # we check the validation loss instead of training loss. OK~
        if saved_total_loss >= val_total_losses.avg:
            saved_total_loss = val_total_losses.avg
            torch.save(model.state_dict(), args.save_path + "/best"+".pth")
            print("\t\tBest Weights updated for decreased validation loss\n")

        else:
            print("\t\tWeights Not updated for undecreased validation loss\n")
        with torch.no_grad():
            test(args.save_path,args.TEST_PATH,t,False,None)
        


        #schdule the learning rate
        scheduler.step(val_total_losses.avg)


    print("*********Finish Training********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
