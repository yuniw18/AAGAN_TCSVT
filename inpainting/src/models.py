import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from .networks import InpaintGenerator, EdgeGenerator, Discriminator,Discriminator_edge,UNET_Discriminator,ASPP_Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')
    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator_edge(in_channels=2,use_sigmoid=config.GAN_LOSS!='hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)


        self.config.GAN_loss = config.GAN_LOSS


        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real,True,True)
        dis_fake_loss = self.adversarial_loss(dis_fake,False,True)
        dis_loss +=(dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake,True,False)
        gen_loss += gen_gan_loss

        # generator feature matching loss

        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) 
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        
        if (config.ASPP == True) and (config.USE_SPEC == True):
            print("Use ASPP discriminator")
            discriminator = ASPP_Discriminator(in_channels=3,use_spectral_norm=True,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')))
        elif (config.ASPP == True) and (config.USE_SPEC == False):
            discriminator = ASPP_Discriminator(in_channels=3,use_spectral_norm=False,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')))
        
        elif (config.UNET == True) and (config.USE_SPEC == True):
            print("Use UNET discriminator")
            discriminator = UNET_Discriminator(in_channels=3,use_spectral_norm=True,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')),GAN_loss=self.config.GAN_loss)
        elif (config.UNET == True) and (config.USE_SPEC == False):
            discriminator = UNET_Discriminator(in_channels=3,use_spectral_norm=False,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')),GAN_loss = self.config.GAN_loss)
        elif (config.Auto == True) and (config.USE_SPEC == True):
            discriminator = Discriminator(in_channels=3,use_spectral_norm=True,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')))
        elif (config.Auto == True) and (config.USE_SPEC == False):
             discriminator = Discriminator(in_channels=3,use_spectral_norm=False,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')))
        elif (config.Auto == False) and (config.USE_SPEC == True):
             discriminator = Discriminator_edge(in_channels=3,use_spectral_norm=True,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')))
        elif (config.Auto == False) and (config.USE_SPEC == False):
              discriminator = Discriminator_edge(in_channels=3,use_spectral_norm=False,use_sigmoid=((self.config.GAN_loss !='hinge') or (self.config.GAN_loss !='rasgan_aver_hinge')))
        else: 
            print("Wrong Discriminator configuration !!")


        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)
        
        self.k = 0
        l1_loss = nn.L1Loss()
        if self.config.USE_percep:
            perceptual_loss = PerceptualLoss()
            style_loss = StyleLoss()
            self.add_module('perceptual_loss', perceptual_loss)
            self.add_module('style_loss', style_loss)
 
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
 
        self.config.GAN_loss = config.GAN_LOSS
        self.batch_size = config.BATCH_SIZE

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        if config.GAN_LOSS == 'wgan_GP':
            self.lambda_ = 10


        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        if self.config.GAN_loss == 'UNET':
            dis_real_dec, dis_real_enc = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake_dec, dis_fake_enc = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        elif self.config.GAN_loss == 'rs_proposed_unet':
            dis_real, dis_real_enc = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake, dis_fake_enc = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        elif self.config.GAN_loss == 'rs_proposed_sample':
            dis_real, dis_real_enc = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake, dis_fake_enc = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        elif self.config.GAN_loss == 'rs_proposed_sample_noavg':
            dis_real, dis_real_enc = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake, dis_fake_enc = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        else:
            dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]

################################# Calculate discriminator loss ############################
        if (self.config.GAN_loss == 'hinge') or (self.config.GAN_loss == 'nsgan') or (self.config.GAN_loss == 'lsgan'):
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) 

        elif self.config.GAN_loss == 'UNET':
            dis_real_loss_enc = self.adversarial_loss(dis_real_enc, True, True)
            dis_fake_loss_enc = self.adversarial_loss(dis_fake_enc, False, True)
            dis_real_loss_dec = self.adversarial_loss(dis_real_dec, True, True)
            dis_fake_loss_dec = self.adversarial_loss(dis_fake_dec, False, True)
            
            dis_loss += (dis_real_loss_enc + dis_fake_loss_enc + dis_real_loss_dec + dis_fake_loss_dec) 
        
        elif self.config.GAN_loss == 'proposed':

            zero = torch.zeros_like(dis_real,requires_grad=False)            
            dis_real_loss = torch.mean(torch.abs((dis_real - zero)))

            sel_images = dis_fake * images + (1.0 - dis_fake) * outputs.detach()
            dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
            disparity = torch.abs(images - outputs.detach())
            disp_loss = torch.mean(torch.abs((dis_fake - self.config.DISP_WEIGHT * disparity))) 

            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss  
            + self.config.DISP_LOSS_WEIGHT * disp_loss) 

        elif self.config.GAN_loss == 'rs_proposed':         
            
            zero = torch.zeros_like(dis_real,requires_grad=False)            
            one = torch.ones_like(dis_fake,requires_grad = False)

            dis_real = torch.nn.ReLU()(1.0 + (dis_real - torch.mean(dis_fake)))
            dis_real_loss = torch.mean(torch.abs((dis_real - zero)))
            
            dis_fake_sel = torch.nn.ReLU()(dis_fake - torch.mean(dis_real))
            dis_fake_sel_loss = torch.mean(torch.abs(dis_fake_sel - one))

            sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
            dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
            disparity = torch.abs(images - outputs.detach())
            disp_loss = torch.mean(torch.abs((dis_fake - self.config.DISP_WEIGHT * disparity)))

            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss 
            + self.config.DISP_LOSS_WEIGHT * disp_loss)

        elif self.config.GAN_loss == 'rs_proposed_noavg':
            
            zero = torch.zeros_like(dis_real,requires_grad=False)            
            one = torch.ones_like(dis_fake,requires_grad = False)

            dis_real = torch.nn.ReLU()(1.0 + (dis_real - dis_fake))
            dis_real_loss = torch.mean(torch.abs((dis_real - zero)))
            
            dis_fake_sel = torch.nn.ReLU()(dis_fake - dis_real)
            dis_fake_sel_loss = torch.mean(torch.abs(dis_fake_sel - one))

            sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
            dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
            disparity = torch.abs(images - outputs.detach())
            disp_loss = torch.mean(torch.abs((dis_fake - self.config.DISP_WEIGHT * disparity)))

            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss 
            + self.config.DISP_LOSS_WEIGHT * disp_loss)



        elif self.config.GAN_loss == 'rs_proposed_unet':
            
            zero = torch.zeros_like(dis_real,requires_grad=False)            
            one = torch.ones_like(dis_fake,requires_grad = False)

            dis_real = torch.nn.ReLU()(1.0 + (dis_real - torch.mean(dis_fake)))
            dis_real_loss = torch.mean(torch.abs((dis_real - zero)))
            
            dis_fake_sel = torch.nn.ReLU()(dis_fake - torch.mean(dis_real))
            dis_fake_sel_loss = torch.mean(torch.abs(dis_fake_sel - one))

            sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
            dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
            disparity = torch.abs(images - outputs.detach())
            disp_loss = torch.mean(torch.abs((dis_fake - self.config.DISP_WEIGHT * disparity)))
            dis_real_loss_enc = self.adversarial_loss(dis_real_enc, False, True)
            dis_fake_loss_enc = self.adversarial_loss(dis_fake_enc, True, True)
 
            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss 
            + self.config.DISP_LOSS_WEIGHT * disp_loss)

            dis_loss += (dis_real_loss_enc + dis_fake_loss_enc)

        elif self.config.GAN_loss == 'rs_proposed_sample_noavg':
            
            zero = torch.zeros_like(dis_real,requires_grad=False)            
            one = torch.ones_like(dis_fake,requires_grad = False)

#           normal relativistic loss is not used for fake part since it is considered when calculating the loss    

            dis_real = torch.nn.ReLU()(1.0 + (dis_real - dis_fake))
            dis_real_loss = torch.mean(torch.abs((dis_real - zero)))
            
            dis_fake_sel = torch.nn.ReLU()(dis_fake - dis_real)
 
            dis_fake_sel_loss = torch.mean(torch.abs(dis_fake_sel - one))

            sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
            dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
            disparity = torch.abs(images - outputs.detach())
            disp_loss = torch.mean(torch.abs((dis_fake - self.config.DISP_WEIGHT * disparity)))

################################# encoder loss part ########################################
            zero = torch.zeros_like(dis_real_enc,requires_grad=False)            
            one = torch.ones_like(dis_fake_enc,requires_grad = False)


            dis_real_enc = torch.nn.ReLU()(1.0 + (dis_real_enc - torch.mean(dis_fake_enc)))
            dis_real_loss_enc = torch.mean(torch.abs((dis_real_enc - zero)))
            
            dis_fake_sel_enc = torch.nn.ReLU()(dis_fake_enc - torch.mean(dis_real_enc))
            dis_fake_sel_loss_enc = torch.mean(torch.abs(dis_fake_sel_enc - one))

            dis_fake_sel_enc = dis_fake_sel_enc.unsqueeze(1).unsqueeze(1)
            sel_images_enc = dis_fake_sel_enc * images + (1.0 - dis_fake_sel_enc) * outputs.detach()
            dis_fake_loss_enc = torch.mean(torch.abs((sel_images_enc - images)))
             
            disparity_enc = torch.mean(torch.abs(images - outputs.detach()))
            disp_loss_enc = torch.mean(torch.abs((torch.mean(dis_fake_enc) - self.config.DISP_WEIGHT * disparity_enc)))

 
            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss 
            + self.config.DISP_LOSS_WEIGHT * disp_loss)

            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss_enc  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss_enc 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss_enc 
            + self.config.DISP_LOSS_WEIGHT * disp_loss_enc)
        
        elif self.config.GAN_loss == 'rs_proposed_sample':
            
            zero = torch.zeros_like(dis_real,requires_grad=False)            
            one = torch.ones_like(dis_fake,requires_grad = False)

#           normal relativistic loss is not used for fake part since it is considered when calculating the loss    

#            dis_real = torch.nn.ReLU()(1.0 + (dis_real - torch.mean(dis_fake)))
            dis_real = dis_real
            dis_real_loss = torch.mean(torch.abs((dis_real - zero)))
            
            dis_fake_sel = dis_fake
#            dis_fake_sel = torch.nn.ReLU()(dis_fake - torch.mean(dis_real))
            dis_fake_sel_loss = torch.mean(torch.abs(dis_fake_sel - one))

            sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
            dis_fake_loss = torch.mean(torch.abs((sel_images - images)))
             
            disparity = torch.abs(images - outputs.detach())
            disp_loss = torch.mean(torch.abs((dis_fake - self.config.DISP_WEIGHT * disparity)))

################################# encoder loss part ########################################
            zero = torch.zeros_like(dis_real_enc,requires_grad=False)            
            one = torch.ones_like(dis_fake_enc,requires_grad = False)


            dis_real_enc = dis_real_enc
#            dis_real_enc = torch.nn.ReLU()(1.0 + (dis_real_enc - torch.mean(dis_fake_enc)))
            dis_real_loss_enc = torch.mean(torch.abs((dis_real_enc - zero)))
            
            dis_fake_sel_enc = dis_fake_enc
#            dis_fake_sel_enc = torch.nn.ReLU()(dis_fake_enc - torch.mean(dis_real_enc))
            dis_fake_sel_loss_enc = torch.mean(torch.abs(dis_fake_sel_enc - one))

            dis_fake_sel_enc = dis_fake_sel_enc.unsqueeze(1).unsqueeze(1)
            sel_images_enc = dis_fake_sel_enc * images + (1.0 - dis_fake_sel_enc) * outputs.detach()
            dis_fake_loss_enc = torch.mean(torch.abs((sel_images_enc - images)))
             
            disparity_enc = torch.mean(torch.abs(images - outputs.detach()))
            disp_loss_enc = torch.mean(torch.abs((torch.mean(dis_fake_enc) - self.config.DISP_WEIGHT * disparity_enc)))

 
            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss 
            + self.config.DISP_LOSS_WEIGHT * disp_loss)

            dis_loss += (self.config.REAL_LOSS_WEIGHT * dis_real_loss_enc  
            + self.config.FAKE_LOSS_WEIGHT * dis_fake_loss_enc 
            + self.config.FAKE_REL_LOSS_WEIGHT * dis_fake_sel_loss_enc 
            + 0 * disp_loss_enc)
            
        elif self.config.GAN_loss == 'rasgan_aver_lsgan':
            dis_real_loss = torch.mean((dis_real - torch.mean(dis_fake) - 1.0)**2) 
            dis_fake_loss = torch.mean((dis_fake - torch.mean(dis_real) + 1.0)**2)

            dis_loss += dis_fake_loss + dis_real_loss

##################################### Calculate generator loss #####################################

        outputs = self(images, edges, masks)
        if (self.config.GAN_loss == 'UNET') :
            gen_fake_dec, gen_fake_enc = self.discriminator(outputs)                    # in: [rgb(3)]
        elif (self.config.GAN_loss == 'rs_proposed_unet') :
            gen_fake, gen_fake_enc = self.discriminator(outputs)                    # in: [rgb(3)]
        elif (self.config.GAN_loss == 'rs_proposed_sample') :
            gen_fake, gen_fake_enc = self.discriminator(outputs)                    # in: [rgb(3)]
        elif (self.config.GAN_loss == 'rs_proposed_sample_noavg') :
            gen_fake, gen_fake_enc = self.discriminator(outputs)                    # in: [rgb(3)]
        else:
            gen_fake, _ = self.discriminator(outputs)                    # in: [rgb(3)]
            gen_real, _ = self.discriminator(images) 

        if self.config.GAN_loss == 'proposed':

            gen_gan_loss = torch.mean(torch.abs((gen_fake)))
            gen_loss += gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
 
        elif self.config.GAN_loss == 'rs_proposed':

            gen_gan_loss = torch.mean(torch.abs((gen_fake)))
            gen_loss += gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
        
        elif self.config.GAN_loss == 'rs_proposed_noavg':

            gen_gan_loss = torch.mean(torch.abs((gen_fake)))
            gen_loss += gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT
        
        elif self.config.GAN_loss == 'rs_proposed_unet':
            gen_gan_loss_enc = self.adversarial_loss(gen_fake_enc,False,False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_gan_loss = torch.mean(torch.abs((gen_fake)))
            gen_loss += (gen_gan_loss + gen_gan_loss_enc) * self.config.INPAINT_ADV_LOSS_WEIGHT
#            gen_loss += (gen_gan_loss) * self.config.INPAINT_ADV_LOSS_WEIGHT
        elif self.config.GAN_loss == 'rs_proposed_sample':
            gen_gan_loss_enc = torch.mean(torch.abs((gen_fake_enc)))
            gen_gan_loss = torch.mean(torch.abs((gen_fake)))
            gen_loss += (gen_gan_loss + gen_gan_loss_enc) * self.config.INPAINT_ADV_LOSS_WEIGHT
        elif self.config.GAN_loss == 'rs_proposed_sample_noavg':
            gen_gan_loss_enc = torch.mean(torch.abs((gen_fake_enc)))
            gen_gan_loss = torch.mean(torch.abs((gen_fake)))
            gen_loss += (gen_gan_loss + gen_gan_loss_enc) * self.config.INPAINT_ADV_LOSS_WEIGHT
        
        elif (self.config.GAN_loss == 'hinge') or (self.config.GAN_loss == 'nsgan') or (self.config.GAN_loss == 'lsgan'):
            gen_gan_loss = self.adversarial_loss(gen_fake,True,False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss +=gen_gan_loss

        elif self.config.GAN_loss == 'UNET':
            gen_gan_loss_enc = self.adversarial_loss(gen_fake_enc,True,False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_gan_loss_dec = self.adversarial_loss(gen_fake_dec,True,False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_gan_loss = ( gen_gan_loss_enc + gen_gan_loss_dec)

            gen_loss += gen_gan_loss


        elif self.config.GAN_loss == 'rasgan_aver_lsgan':
            gen_real, _ = self.discriminator(images) 
            gen_gan_loss = torch.mean((gen_real - torch.mean(gen_fake) + 1.0)**2) + torch.mean((gen_fake - torch.mean(gen_real) - 1.0) ** 2)
            gen_loss += gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        if self.config.USE_percep:
            gen_content_loss = self.perceptual_loss(outputs, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss


#       generator style loss
            gen_style_loss = self.style_loss(outputs * masks, images * masks)
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss

        # create logs
        if self.config.USE_percep:
            logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
            ]
        else:
            logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ]
 
        return outputs, gen_loss, dis_loss, logs,gen_gan_loss

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        
        dis_loss.backward() 
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


