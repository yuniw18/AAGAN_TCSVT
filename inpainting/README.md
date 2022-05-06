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


This repository composed of Three parts
* 1. How to implement to your own network.
* 2. How to reproduce results of the paper.
* 3. Demo


## 1. How to implement to your own network

### prerequisites

Because our method is about a loss function, it is easily applicable to any other method if conditions below are met.

* Auto-encoder based discriminator
* Conditional Image generation tasks where ground truth exists, e.g. inpainting.

Since our method uses pixel-wise differences between samples, image generation tasks where fitting data distribution(i.e., unsupervised learning) only is not appropriate.
If your project does not contain the auto-encoder discriminator, you can just use the 'Discriminator' in `networks.py`
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


## 2. How to reproduce results of the paper

To prove effectiveness & aplicability, we applied an AAGAN to Inpainting, Super resolution and Frame interpolation.
For inpainting, we used [Edge-connect](https://github.com/knazeri/edge-connect) as a experiment environment
For frame interpolation, we used [DAIN](https://github.com/baowenbo/DAIN).
We will cover inpainting only for this repository, however, it can be easily applicable to other methods in a similar manner.
Note that, this repository is highly based on [Edge-connect](https://github.com/knazeri/edge-connect). 
Therefore, refer to Edge-connect about details regarding installation. 

### prerequisites
* refer to [Edge-connect](https://github.com/knazeri/edge-connect).
### Installation
* clone the repository
~~~
git clone ...
~~~
* Install dependencies (refer to [Edge-connect](https://github.com/knazeri/edge-connect)).

### Datasets
#### 1) images

We used [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|[Places2](http://places2.csail.mit.edu/) dataset.

#### 2) masks
We used [testing irregular mask datasets](http://masc.cs.gmu.edu/wiki/partialconv) for training/testing mask dataset without distinction.

#### 3) Pre-processing
Details about pre-processing dataset can be found in [Edge-connect](https://github.com/knazeri/edge-connect).

### Getting Started
#### 1) Set model configuration
Most of the configuration is same with [Edge-connect](https://github.com/knazeri/edge-connect).

* **USE_SPEC**: *True* -> use spectral normalization for discriminator. *False* -> None.
* **AUTO**: *True* -> Use auto-encoder based discriminator. *False* -> Use normal discriminator.
* **GAN_LOSS**: 
                *hinge,nsgan,lsgan* -> set loss function according to each configuration.  
                *rasgan_aver_lsgan* -> set loss function with ragan which uses least square error.                
                *proposed* -> set loss function with AAGAN.  
                *rs_proposed* -> set loss function with RAAGAN.  


For example, if configuration is set as **USE_SPEC** : *True*, **AUTO** : *False* and  **GAN_LOSS** : *lsgan*, GAN loss will be least square error with spectral normalization(SN-lsgan).

#### 2) Training
We used pre-trained model for fast convergence. Download pre-trained model at [Edge-connect](https://github.com/knazeri/edge-connect).
~~~bash
python3 train.py --model 2 --checkpoints [checkpoint_path]
~~~

#### 3) Testing
~~~bash
python3 test.py --model 3 --checkpoints [checkpoint_path]
~~~

#### 4) Measure Quantitive results.
* Objective Quality

* Subjective Quality

## 3. Demo
Here, we will explain how to execute simple demo above.

The images in demo_image is cropped from PCGAN's official youtube. [Paper](https://arxiv.org/abs/1710.10196)|[code](https://github.com/tkarras/progressive_growing_of_gans)|[presentation(youtube)](https://youtu.be/G06dEcZ-QTg)
The pre-trained model in checkpoints/Sample is trained using parameters at Table 2, edge-connect* in Technical Appendix.
EdgeModel_gen.pth is trained in advance of Inpainting model.
Follow this instruction.

1. Prepare data file list
Make image flist.
~~~bash
python3 scripts/flist.py --path ./demo_image/ --output ./datasets/demo.flist
~~~
Make mask flist(must be downloaded first from the above path).
~~~bash
python3 scripts/flist.py --path [Mask_path] --output ./datasets/MASK.flist
~~~
2. Extract image from the pre-trained model.
~~~bash
python3 test.py --model 3 --checkpoints ./checkpoints/Sample
~~~

Then, the results will be saved in checkpoints/Sample/results folder. 

## Citation
Will be updated soon.

## License
Our contributions on codes are released under the MIT license. For the codes of the otehr works, refer to their repositories.


