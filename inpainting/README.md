# Inpainting

This repository contains the code of the AAGAN for inpainting.

## 1. How to reproduce results of the paper

For inpainting, we used [Edge-connect](https://github.com/knazeri/edge-connect) as a baseline.
Refer to [Edge-connect](https://github.com/knazeri/edge-connect) for details regarding installation. 
Experiment environment 

### Installation
* Install dependencies by referring to [Edge-connect](https://github.com/knazeri/edge-connect) or **environment.yml** ).

### Prepare datasets
#### 1) images

We used [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|[Places2](http://places2.csail.mit.edu/) dataset.

#### 2) masks
We used [testing irregular mask datasets](http://masc.cs.gmu.edu/wiki/partialconv) for training/testing mask dataset without distinction.

#### 3) Pre-processing
Details about pre-processing dataset can be found in [Edge-connect](https://github.com/knazeri/edge-connect).

## 2. Quick start
Here, we explain how to execute simple demo.

The images in **demo_image** folder is cropped from PCGAN's official youtube. [Paper](https://arxiv.org/abs/1710.10196)|[code](https://github.com/tkarras/progressive_growing_of_gans)|[presentation(youtube)](https://youtu.be/G06dEcZ-QTg).


1. Prepare data file list
Make image flist.
~~~bash
python3 scripts/flist.py --path ./demo_image/ --output ./datasets/demo.flist
~~~
Make mask flist(download them from the link above).
~~~bash
python3 scripts/flist.py --path [Mask_path] --output ./datasets/MASK.flist
~~~
2. Extract image from the pre-trained model.
~~~bash
python3 test.py --model 3 --checkpoints ./checkpoints/Sample
~~~

Then, the results will be saved in checkpoints/Sample/results folder. 


## 3. Getting Started
### 1) Set model configuration
Most of the configuration is same with [Edge-connect](https://github.com/knazeri/edge-connect).

* **USE_SPEC**: *True* -> use spectral normalization for discriminator. *False* -> None.
* **AUTO**: *True* -> Use auto-encoder based discriminator. *False* -> Use normal discriminator.
* **GAN_LOSS**: 
                *hinge,nsgan,lsgan* -> set loss function according to each configuration.  
                *rasgan_aver_lsgan* -> set loss function with ragan which uses least square error.                
                *proposed* -> set loss function with AAGAN.  
                *rs_proposed* -> set loss function with RAAGAN.  


For example, if configuration is set as **USE_SPEC** : *True*, **AUTO** : *False* and  **GAN_LOSS** : *lsgan*, GAN loss will be least square error with spectral normalization(SN-lsgan). Sample configuration is included in **./checkpoints/SAMPLE/config.yml**

### 2) Training
We used pre-trained model for fast convergence. Download pre-trained model at [Edge-connect](https://github.com/knazeri/edge-connect).
~~~bash
python3 train.py --model 2 --checkpoints [checkpoint_path]
~~~

### 3) Testing
~~~bash
python3 test.py --model 3 --checkpoints [checkpoint_path]
~~~


## License
Our contributions on codes are released under the MIT license. For the codes of the otehr works, refer to their repositories.


