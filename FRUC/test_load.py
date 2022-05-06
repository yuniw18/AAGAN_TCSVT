import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import scipy
import numpy as np
import random
import scipy
#def UCF_loader(root, im_path, input_frame_size = (3, 256, 256), output_frame_size = (3, 256, 256), data_aug = False,index=0):

class TestDataset(data.Dataset):
    def __init__(self, root,input_frame_size = (3, 256, 448), output_frame_size = (3, 240, 240)):
                
        dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
        dir_path.sort()     
        self.image_paths=[]
        self.input_frame_size = input_frame_size
        self.output_frame_size = output_frame_size
        for dir_name in dir_path:
            image_name = os.listdir(dir_name)
            image_name = [f for f in image_name if f.endswith('png')]
            image_name.sort()
            for name in image_name:
                self.image_paths.append(os.path.join(dir_name,name))         
 
    def __getitem__(self,index):
        im_size = (240,320,3)
        path_pre1 = self.image_paths[index]
        path_pre2 = self.image_paths[index + 2]
        path_mid = self.image_paths[index + 1]
 
        im_pre2 = scipy.misc.imresize(imread(path_pre2),im_size)
        im_pre1 = scipy.misc.imresize(imread(path_pre1),im_size)
        im_mid = scipy.misc.imresize(imread(path_mid),im_size)

        im_pre2 = np.pad(im_pre2,((8,8),(64,64),(0,0)),'constant')
        im_pre1 = np.pad(im_pre1,((8,8),(64,64),(0,0)),'constant')
        im_mid = np.pad(im_mid,((8,8),(64,64),(0,0)),'constant')
       

        h_offset = random.choice(range(256 - self.input_frame_size[1] + 1))
        w_offset = random.choice(range(448 - self.input_frame_size[2] + 1))

#        h_offset = 30
#        w_offset = 0

        im_pre2 = im_pre2[h_offset:h_offset + self.input_frame_size[1], w_offset: w_offset + self.input_frame_size[2], :]
        im_pre1 = im_pre1[h_offset:h_offset + self.input_frame_size[1], w_offset: w_offset + self.input_frame_size[2], :]
        im_mid = im_mid[h_offset:h_offset + self.input_frame_size[1], w_offset: w_offset + self.input_frame_size[2], :]
#        im_pre2 = im_pre2[h_offset:h_offset + 186, w_offset: w_offset + self.input_frame_size[2], :]
#        im_pre1 = im_pre1[h_offset:h_offset + 186, w_offset: w_offset + self.input_frame_size[2], :]
#        im_mid = im_mid[h_offset:h_offset + 186, w_offset: w_offset + self.input_frame_size[2], :]



        X0 = np.transpose(im_pre1,(2,0,1))
        X2 = np.transpose(im_pre2, (2, 0, 1))

        y = np.transpose(im_mid, (2, 0, 1))

        return X0.astype("float32")/ 255.0, \
            X2.astype("float32")/ 255.0,\
            y.astype("float32")/ 255.0
    def __len__(self):
        return len(self.image_paths)


#class TestDataset(data.Dataset):
#    def __init__(self, root, path_list,  loader=UCF_loader):

#        self.root = root
#        self.path_list = path_list
#        self.loader = loader

#    def __getitem__(self, index):
#        path = self.path_list[index]
        # print(path)
#        image_0,image_2,image_1 = self.loader(self.root, None,index=index)
#        return image_0,image_2,image_1
#
#    def __len__(self):
#        return len(self.path_list)
