import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
    
        # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

#        self.conv_1x1_4 = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # feature map의 shape은 (batch_size, in_channels, height/output_stride, width/output_stride)

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        # shape: (batch_size, in_channels, 1, 1)
        out_img = self.avg_pool(feature_map) 
        # shape: (batch_size, out_channels, 1, 1)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        # shape: (batch_size, out_channels * 5, height/output_stride, width/output_stride)
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 
        # shape: (batch_size, num_classes, height/output_stride, width/output_stride)
#        out = self.conv_1x1_4(out) 

        return out

class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
 
        
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.InstanceNorm2d(out_channels*2, track_running_stats=False),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation),
                                                                    nn.InstanceNorm2d(out_channels, track_running_stats=False),
 ))

    def forward(self, x):
        return self.atrous_conv.forward(x)

#class atrous_conv(nn.Sequential):
#    def __init__(self, in_channels, out_channels, dilation,spec = True):
#        super(atrous_conv, self).__init__()
#        self.atrous_conv = torch.nn.Sequential()
 
        
#        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(
#                                                                    spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2, bias=False, kernel_size=1, stride=1, padding=0)),
##                                                                    nn.InstanceNorm2d(out_channels, track_running_stats=False),
#                                                                    nn.ReLU(),
#                                                                    spectral_norm(nn.Conv2d(in_channels=out_channels* 2 , out_channels=out_channels, bias=False, kernel_size=3, stride=1,
#                                                                              padding=(dilation, dilation), dilation=dilation)),
#                                                                    nn.InstanceNorm2d(out_channels, track_running_stats=False),
#                                                                    nn.ReLU()
# ))

#    def forward(self, x):
#        return self.atrous_conv.forward(x)
 
class atrous_conv_bn(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)
 
       
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

class Discriminator(BaseNetwork):
    def __init__(self, in_channels,residual_blocks=0,use_spectral_norm=False, init_weights=True,use_sigmoid=False):
        super(Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
#            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )
        self.linear = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU() 
        if init_weights:
            self.init_weights()
        
    def forward(self, x):
        conv = self.encoder(x)
        
        enc_out = conv
        bout = torch.sum(conv, [2, 3])
        bout = self.linear(bout.view(-1,256))
        enc_out = torch.sigmoid(bout)
       
        conv = self.relu(conv) 
        res = self.middle(conv)
        x = self.decoder(res)
        x = (torch.tanh(x) + 1) / 2

        return x,enc_out

class ASPP_Discriminator(BaseNetwork):
    def __init__(self, in_channels,residual_blocks=0,use_spectral_norm=False, init_weights=True,use_sigmoid=False):
        super(ASPP_Discriminator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(32, track_running_stats=False),
            nn.ReLU(True),
            )

        self.encoder2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
#            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),use_spectral_norm),
#            nn.InstanceNorm2d(256, track_running_stats=False),
#            nn.ReLU(True)
        )

        in_f = 64  
        out_f = 32
        self.daspp_3    = atrous_conv(in_f, out_f, 3)
        self.daspp_6    = atrous_conv(in_f  ,out_f, 6)
        self.daspp_12   = atrous_conv(in_f, out_f, 12)
        self.daspp_18   = atrous_conv(in_f , out_f, 18)
        self.daspp_24   = atrous_conv(in_f, out_f, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(in_f + out_f * 5 ,in_f, 3, 1, 1, bias=False),
                                              nn.ReLU())



        self.decoder1 = nn.Sequential(
#            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
#            nn.InstanceNorm2d(128, track_running_stats=False),
#            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(32, track_running_stats=False),
            nn.ReLU(True),
        )
 
        self.decoder2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=7,  padding=0),
        )

        self.relu = nn.ReLU() 
        if init_weights:
            self.init_weights()
        
    def forward(self, x):
        enc_out = 1

        conv1 = self.encoder1(x)
        x = self.encoder2(conv1)  
        

        daspp_3 = self.daspp_3(x)
        daspp_6 = self.daspp_6(x)
        daspp_12 = self.daspp_12(x)
        daspp_18 = self.daspp_18(x)
        daspp_24 = self.daspp_24(x)
        concat4_daspp = torch.cat([x,daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        
        x = self.daspp_conv(concat4_daspp)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = (torch.tanh(x) + 1) / 2
        return x,enc_out

class ASPP_Discriminator_ASPP(BaseNetwork):
    def __init__(self, in_channels,residual_blocks=0,use_spectral_norm=False, init_weights=True,use_sigmoid=False):
        super(ASPP_Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )


        in_f = 256  
        out_f = 256
        self.ASPP = ASPP(in_f,out_f)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7,  padding=0),
        )

        self.relu = nn.ReLU() 
        if init_weights:
            self.init_weights()
        
    def forward(self, x):
        enc_out = 1
        x = self.encoder(x)

        x = self.ASPP(x)

        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x,enc_out


class ASPP_Discriminator_ori(BaseNetwork):
    def __init__(self, in_channels,residual_blocks=0,use_spectral_norm=False, init_weights=True,use_sigmoid=False):
        super(ASPP_Discriminator_ori, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(32, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
#            spectral_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
#            nn.InstanceNorm2d(256, track_running_stats=False),
#            nn.ReLU(True)
        )
        
        self.daspp_3    = atrous_conv(64, 32, 3)
        self.daspp_6    = atrous_conv(64 ,32, 6)
        self.daspp_12   = atrous_conv(64 , 32, 12)
        self.daspp_18   = atrous_conv(64 , 32, 18)
        self.daspp_24   = atrous_conv(64 , 32, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(64 + 32 * 5,64, 3, 1, 1, bias=False),
                                              nn.ReLU())
        self.decoder = nn.Sequential(
#            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
#            nn.InstanceNorm2d(128, track_running_stats=False),
#            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(32, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=7,  padding=0),
        )

        self.relu = nn.ReLU() 
        if init_weights:
            self.init_weights()
        
    def forward(self, x):
        enc_out = 1
        x = self.encoder(x)
        daspp_3 = self.daspp_3(x)
#        concat4_2 = torch.cat([x, daspp_3], dim=1)
        daspp_6 = self.daspp_6(x)
#        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(x)
#        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(x)
#        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(x)
        concat4_daspp = torch.cat([x,daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        x = self.daspp_conv(concat4_daspp)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x,enc_out


class UNET_Discriminator(BaseNetwork):
    def __init__(self, in_channels,residual_blocks=0,use_spectral_norm=False, init_weights=True,use_sigmoid=False,GAN_loss='UNET'):
        super(UNET_Discriminator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
            )
#4
        self.encoder2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            )
#2
        self.encoder3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
#            nn.ReLU(True)
        )
#1
        self.relu = nn.ReLU(True)    
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        self.decoder1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            )

        self.decoder2 = nn.Sequential(
    
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )
        
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=7, padding=0),
        )

        self.linear = nn.Linear(256, 1)
        self.GAN_loss = GAN_loss
        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.encoder1(x)
        conv2 = self.encoder2(conv1)  
        conv3 = self.encoder3(conv2)
        
        if self.GAN_loss == 'UNET':
            enc_out = conv3 
        else:
            enc_out = conv3
        conv3 = self.relu(conv3)
        
        deconv1 = self.decoder1(conv3)
        u1 = torch.cat((deconv1,conv2), dim=1)
        deconv2 = self.decoder2(u1)
        u2 = torch.cat((deconv2,conv1),dim=1)
        decoder3 = self.decoder3(u2)
        
        if self.GAN_loss == 'UNET':
            dec_out = decoder3        
        else:
            dec_out = (torch.tanh(decoder3) + 1) / 2

        return dec_out, enc_out


class Discriminator_edge(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=False, init_weights=True):
        super(Discriminator_edge, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
