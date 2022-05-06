# AAGAN_DAIN

This repository is an AAGAN, which are imported in [DAIN](https://github.com/baowenbo/DAIN).

Since this code is highly based on [DAIN](https://github.com/baowenbo/DAIN), please refer to their project page for prerequisites or installation. 

## How to reproduce results in a paper

### 1. Datasets
#### 1) Training datasets
DAIN uses Vimeo 90k datasets for training. Check here to download [Vimeo 90k](https://github.com/anchen1011/toflow/blob/master/download_dataset.sh)

#### 2) Test datasets
We use UCF-101 and Middlebury dataset for test. 
To download Middlebury set, prepare folder and do:
~~~bash
wget http://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
unzip other-color-allframes.zip
wget http://vision.middlebury.edu/flow/data/comp/zip/other-gt-interp.zip
unzip other-gt-interp.zip
~~~
To download UCF-101 dataset, please refer to [here](https://github.com/harvitronix/five-video-classification-methods).

### 2. Getting Started
#### 1) Set model configuration

* **MULTI_FRAME**: *True* -> use reference frame along with target frame for discriminator input. We use True for this setting in our paper *False* -> None.
* **AUTO**: *True* -> Use auto-encoder based discriminator. *False* -> Use normal discriminator.
* **GAN_LOSS**: 
                *rasgan_aver_lsgan* -> set loss function with ragan which uses least square error.                
                *proposed* -> set loss function with AAGAN.          
For other parameters, please refer to **my_args.py**.

#### 2) Training & Testing
To train DAIN with AAGAN do:

~~~bash
python3 train.py --pretrained pretrained_model --uid proposed --datasetPath ./vimeo_dataset/vimeo_triplet/ --rectify_lr 0.0001 --AUTO True --MULTI_FRAME True
~~~

For each epochs, the model will do validation/test on Vimeo 90k dataset and Middlebury dataset.
To test DAIN on UCF dataset, do:
~~~bash
python3 test.py --pretrained pretrained_model --uid proposed --datasetPath ./vimeo_dataset/vimeo_triplet/ --rectify_lr 0.0001 --AUTO True --MULTI_FRAME True
~~~
### 3. Demo
Here, We will explain how to execute simple demo.
We provide the pretrained model of AAGAN10, which is used for experiments in a paper.
Please, check the pre_trained folder.
The demo results are generated using those pre-trained model.
To generate results with your own model, modify the argument in **demo_AAGAN.py**.

To reproduce demo results in a paper and Multimedia Appendix, do:
~~~bash
python3 demo_AAGAN.py --netName DAIN_slowmotion --time_step 0.01
~~~
Then, the results will be saved in /demo_set/demo_results.
## Lincense
Refer to [DAIN](https://github.com/baowenbo/DAIN).
## Citation
Refer to [DAIN](https://github.com/baowenbo/DAIN).
