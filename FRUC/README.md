# AAGAN_DAIN

This repository contains the code of FRUC with an AAGAN.

We used [DAIN](https://github.com/baowenbo/DAIN) as a baseline. 

### 1. Instalation

Please refer to [DAIN](https://github.com/baowenbo/DAIN) for details about prerequisites or installation. 

### 2. Datasets
#### 1) Training datasets
We use Vimeo 90k datasets for training. Check here to download [Vimeo 90k](https://github.com/anchen1011/toflow/blob/master/download_dataset.sh)

#### 2) Test datasets
We use UCF-101 and Middlebury dataset additionally for evaluation. 
To download Middlebury set, prepare folder and do:
~~~bash
wget http://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
unzip other-color-allframes.zip
wget http://vision.middlebury.edu/flow/data/comp/zip/other-gt-interp.zip
unzip other-gt-interp.zip
~~~
To download UCF-101 dataset, please refer to [here](https://github.com/harvitronix/five-video-classification-methods).

### 3. Quick start
Here, we explain how to execute simple demo.
We provide the pretrained model of AAGAN_10 in **pre_trained** folder., which is used for experiments in a paper.
To generate results with your own model, modify the argument in **demo_AAGAN.py**.
To reproduce demo results in a paper and Multimedia Appendix, do:

~~~bash
python3 demo_AAGAN.py --netName DAIN_slowmotion --time_step 0.01
~~~
Then, the results will be saved in **./demo_set/demo_results** folder.

### 4. Getting Started
#### 1) Set model configuration
For details, please refer to **my_args.py**.

#### 2) Training & Testing
To train DAIN with an AAGAN, do:

~~~bash
python3 train.py --pretrained pretrained_model --uid proposed --datasetPath ./vimeo_dataset/vimeo_triplet/ --rectify_lr 0.0001 --AUTO True --MULTI_FRAME True
~~~

For each epochs, the model will do validation/test on Vimeo 90k dataset and Middlebury dataset.
To test DAIN on UCF dataset, do:
~~~bash
python3 test.py --pretrained pretrained_model --uid proposed --datasetPath ./vimeo_dataset/vimeo_triplet/ --rectify_lr 0.0001 --AUTO True --MULTI_FRAME True
~~~

## Lincense
Our contributions on codes are released under the MIT license. For the codes of the otehr works, refer to their repositories.

