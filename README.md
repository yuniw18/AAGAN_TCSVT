# AAGAN

This repository contains the code of the paper 
> AAGAN: Accuracy-Aware Generative Adversarial Network for supervised tasks
>
>Ilwi Yun, Hyuk-Jae Lee, Chae Eun Rhee.
>
> IEEE transactions on Circuits and Systems for Video Technology


## Introduction
![demo](./checkpoints/Sample/AAGAN_demo.gif)

* Ground truth: results of PCGAN(cropped from [presentation(youtube)](https://youtu.be/G06dEcZ-QTg),  
* Input: [testing irregular mask datasets](http://masc.cs.gmu.edu/wiki/partialconv),  
* edge-connect: Results when trained using SNGAN,  
* edge-connect*: Results when trained using AAGAN.

The above demo shows some strengths when AAGAN is applied to inpainting. The boundary of masked region is visible in Edge-connect, while rarely visible in Edge-connect*. This implies that discriminator of SNGAN doesn't care about objective quality which changes the oveall tone of the faces as a result. We believe that our work will give an insight to people who studies the image generation area where objective quality should be concerned. 


## 1. How to implement to your own network

### prerequisites

Because our method is about a loss function, it is easily applicable to any other method if conditions below are met.

* Auto-encoder based discriminator
* Conditional Image generation tasks where ground truth exists, e.g. inpainting.

Since our method utilizes pixel-wise differences between samples, image generation tasks of fitting data distribution(i.e., unsupervised learning) only is not appropriate.
Note that, auto-encoder discriminator should output value range from 0 to 1.

### Implementation
To implement an AAGAN to your project, modify the code where GAN loss is caculated as follows.

* Accuracy-aware Discriminator

~~~python
#dis_input_fake : output of generator
#dis_input_real : ground truth
#dis_real,_ = self.discriminator(dis_input_real)
#dis_fake,_ self.discriminator(dis_input_fake)

zero = torch.zeros_like(dis_real,requires_grad=False)            
dis_real_loss = torch.mean(torch.abs((dis_real - zero)))

sel_images = dis_fake * images + (1.0 - dis_fake) * outputs.detach()
dis_fake_loss = torch.mean(torch.abs((sel_images - images)))

disparity = torch.abs(images - outputs.detach())
disp_loss = torch.mean(torch.abs((dis_fake - self.reg * disparity))) # self.reg -> weight of the disparity.

dis_loss += (dis_real_loss  + dis_fake_loss  +  disp_loss)  # dis_loss -> loss of discriminator and should be backward later
~~~
* Relativistic Accuracy-aware Discriminator
~~~python
zero = torch.zeros_like(dis_real,requires_grad=False)
one = torch.ones_like(dis_fake,requires_grad = False)

dis_real = torch.nn.ReLU()(1.0 + (dis_real - torch.mean(dis_fake)))
dis_real_loss = torch.mean(torch.abs((dis_real - zero)))                      # relativistic loss for real part

dis_fake_sel = torch.nn.ReLU()(dis_fake - torch.mean(dis_real))
dis_fake_sel_loss = torch.mean(torch.abs(dis_fake_sel - one))                 # relativistic loss for fake part

sel_images = dis_fake_sel * images + (1.0 - dis_fake_sel) * outputs.detach()
dis_fake_loss = torch.mean(torch.abs((sel_images - images)))                  # lowered criterion part 

disparity = torch.abs(images - outputs.detach())
disp_loss = torch.mean(torch.abs((dis_fake - self.reg * disparity))) # regularization

# Hyper parameters for each losses are ommitted here. Plaese set this manually.
dis_loss += (dis_real_loss  + dis_fake_loss + dis_fake_sel_loss + disp_loss) # dis_loss -> loss of discriminator and should be backward later
~~~

* Generator
~~~python
# outputs : output of generator

gen_fake,_ = self.discriminator(outputs)
gen_gan_loss = torch.mean(torch.abs((gen_fake)))
gen_loss += gen_gan_loss * self.config.INPAINT_ADV_LOSS_WEIGHT

~~~

After caculating loss, backward and optimize according to your code.


## 2. Reproduction

To reproduce the results in the paper, follow the instructions contained in each folder (inpainting, FRUC).


## Citation
Will be updated soon.

## License
Our contributions on codes are released under the MIT license. For the codes of the otehr works, refer to their repositories.
