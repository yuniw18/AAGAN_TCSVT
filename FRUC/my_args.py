import os
import datetime
import argparse
import numpy
import networks
import  torch
modelnames =  networks.__all__
# import datasets
datasetNames = ('Vimeo_90K_interp') #datasets.__all__

parser = argparse.ArgumentParser(description='DAIN')

'''
Added part from the original DAIN code
'''
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser.add_argument('--GAN_loss', type=str,default ='proposed' , help='Set GAN_loss') # proposed -> AAGAN, rs_proposed -> RAAGAN / for others, refer to train.py
parser.add_argument('--USE_SPEC', default =True,type=boolean_string,help='Set Spectral normalization')
parser.add_argument('--Auto', default =True ,type=boolean_string,help='Set discriminator architecture')
parser.add_argument('--Fine_Tuning', type=boolean_string,default =True , help='If true, train only rectifyNet')
parser.add_argument('--D2G_LR', type=float,default =0.1 , help='ratio between Generator and discriminator learning rate')
parser.add_argument('--GAN_LOSS_WEIGHT', type=float,default =1 , help='Set GAN_loss Weight')
parser.add_argument('--MULTI_FRAME', type=boolean_string,default =False , help='Set discriminator to get multiple input frame')
parser.add_argument('--DISP_WEIGHT', type=float,default =1.0 , help='Set discriminator disparity ratio')
parser.add_argument('--FAKE_LOSS_WEIGHT', type=float,default = 1 , help='Set discriminator fake loss ratio')
parser.add_argument('--DISP_LOSS_WEIGHT', type=float,default =1 , help='Set discriminator disparity loss ratio')
parser.add_argument('--FAKE_REL_LOSS_WEIGHT', type=float,default =0 , help='Set discriminator disparity loss ratio')
parser.add_argument('--TESTDATA_PATH', type=str,default = '/hdd/yuniw/UCF/data/test/' , help='Path where test images exist')
parser.add_argument('--test_uid', type=str,default = None , help='Path where test images exist')

parser.add_argument('--debug',action = 'store_true', help='Enable debug mode')
parser.add_argument('--netName', type=str, default='DAIN',
                    choices = modelnames,help = 'model architecture: ' +
                        ' | '.join(modelnames) +
                        ' (default: DAIN)')

parser.add_argument('--datasetName', default='Vimeo_90K_interp',
                    choices= datasetNames,nargs='+',
                    help='dataset type : ' +
                        ' | '.join(datasetNames) +
                        ' (default: Vimeo_90K_interp)')
parser.add_argument('--datasetPath',default='',help = 'the path of selected datasets')
parser.add_argument('--dataset_split', type = int, default=97, help = 'Split a dataset into trainining and validation by percentage (default: 97)')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

parser.add_argument('--numEpoch', '-e', type = int, default=1, help= 'Number of epochs to train(default:150)')

parser.add_argument('--batch_size', '-b',type = int ,default=1, help = 'batch size (default:1)' )
parser.add_argument('--workers', '-w', type =int,default=8, help = 'parallel workers for loading training samples (default : 1.6*10 = 16)')
parser.add_argument('--channels', '-c', type=int,default=3,choices = [1,3], help ='channels of images (default:3)')
parser.add_argument('--filter_size', '-f', type=int, default=4, help = 'the size of filters used (default: 4)',
                    choices=[2,4,6, 5,51]
                    )


parser.add_argument('--lr', type =float, default= 0.002, help= 'the basic learning rate for three subnetworks (default: 0.002)')
parser.add_argument('--rectify_lr', type=float, default=0.001, help  = 'the learning rate for rectify/refine subnetworks (default: 0.001)')

parser.add_argument('--save_which', '-s', type=int, default=1, choices=[0,1], help='choose which result to save: 0 ==> interpolated, 1==> rectified')
parser.add_argument('--time_step',  type=float, default=0.5, help='choose the time steps')
parser.add_argument('--flow_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
parser.add_argument('--occ_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--filter_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--ctx_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--depth_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
# parser.add_argument('--deblur_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')

parser.add_argument('--alpha', type=float,nargs='+', default=[0.0, 1.0], help= 'the ration of loss for interpolated and rectified result (default: [0.0, 1.0])')

parser.add_argument('--epsilon', type = float, default=1e-6, help = 'the epsilon for charbonier loss,etc (default: 1e-6)')
parser.add_argument('--weight_decay', type = float, default=0, help = 'the weight decay for whole network ' )
parser.add_argument('--patience', type=int, default=5, help = 'the patience of reduce on plateou')
parser.add_argument('--factor', type = float, default=0.2, help = 'the factor of reduce on plateou')
#
parser.add_argument('--pretrained', dest='SAVED_MODEL', default=None, help ='path to the pretrained model weights')
parser.add_argument('--no-date', action='store_true', help='don\'t append date timestamp to folder' )
parser.add_argument('--use_cuda', default= True, type = bool, help='use cuda or not')
parser.add_argument('--use_cudnn',default=1,type=int, help = 'use cudnn or not')
parser.add_argument('--dtype', default=torch.cuda.FloatTensor, choices = [torch.cuda.FloatTensor,torch.FloatTensor],help = 'tensor data type ')
# parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')


parser.add_argument('--uid', type=str, default= None, help='unique id for the training')
parser.add_argument('--force', action='store_true', help='force to override the given uid')

args = parser.parse_args()

import shutil

if args.uid == None:
    unique_id = str(numpy.random.randint(0, 100000))
    print("revise the unique id to a random numer " + str(unique_id))
    args.uid = unique_id
    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")
    save_path = '../model_weights/'+ args.uid  +'-' + timestamp
else:
    save_path = '../model_weights/'+ str(args.uid)

# print("no pth here : " + save_path + "/best"+".pth")
if not os.path.exists(save_path + "/best"+".pth"):
    # print("no pth here : " + save_path + "/best" + ".pth")
    os.makedirs(save_path,exist_ok=True)
else:
    if not args.force:
        raise("please use another uid ")
    else:
        print("override this uid" + args.uid)
        for m in range(1,10):
            if not os.path.exists(save_path+"/log.txt.bk" + str(m)):
                shutil.copy(save_path+"/log.txt", save_path+"/log.txt.bk"+str(m))
                shutil.copy(save_path+"/args.txt", save_path+"/args.txt.bk"+str(m))
                break


sample_path = save_path + '/results/'
test_path = save_path + '/benchmark/'
model_path = save_path+ '/best.pth/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

#####################################################
parser.add_argument('--SAMPLE_PATH', default =sample_path , help='Path where sample images will be saved')
parser.add_argument('--TEST_PATH', default=test_path , help='Path where test image will be saved')
parser.add_argument('--MODEL_PATH', default=model_path , help='Path where pretrained model exists')
#####################################################

parser.add_argument('--save_path',default=save_path,help = 'the output dir of weights')
parser.add_argument('--log', default = save_path+'/log.txt', help = 'the log file in training')
parser.add_argument('--arg', default = save_path+'/args.txt', help = 'the args used')

args = parser.parse_args()


with open(args.log, 'w') as f:
    f.close()
with open(args.arg, 'w') as f:
    print(args)
    print(args,file=f)
    f.close()
if args.use_cudnn:
    print("cudnn is used")
    torch.backends.cudnn.benchmark = True  # to speed up the
else:
    print("cudnn is not used")
    torch.backends.cudnn.benchmark = False  # to speed up the

