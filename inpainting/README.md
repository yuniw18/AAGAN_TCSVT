# Inpainting

This repository contains the code of inpainting with an AAGAN.

## 1. Installation

For inpainting, we used [Edge-connect](https://github.com/knazeri/edge-connect) as a baseline.
Refer to [Edge-connect](https://github.com/knazeri/edge-connect) for details abount installation.

### Dependencies
* Install dependencies by referring to [Edge-connect](https://github.com/knazeri/edge-connect) or **environment.yml** .

### Prepare datasets
#### 1) images

We used [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|[Places2](http://places2.csail.mit.edu/) dataset.

#### 2) masks
We used [testing irregular mask datasets](http://masc.cs.gmu.edu/wiki/partialconv) for training/testing mask dataset without distinction.

#### 3) Pre-processing
Details about pre-processing dataset can be found in [Edge-connect](https://github.com/knazeri/edge-connect).

## 2. Quick start
Here, we explain how to execute simple demo, which can reproduce the results in Multimedia Appendix.

The images in **demo_image** folder is extracted from PCGAN's official youtube. [Paper](https://arxiv.org/abs/1710.10196)|[code](https://github.com/tkarras/progressive_growing_of_gans)|[presentation(youtube)](https://youtu.be/G06dEcZ-QTg).


1. Make image flist.
~~~bash
python3 scripts/flist.py --path ./demo_image/ --output ./datasets/demo.flist
~~~
2. Make mask flist (Download the mask data from the link above).
~~~bash
python3 scripts/flist.py --path [Mask_path] --output ./datasets/MASK.flist
~~~
3. Extract image from the pre-trained model.
~~~bash
python3 test.py --model 3 --checkpoints ./checkpoints/Sample
~~~

Then, the results will be saved in **checkpoints/Sample/results** folder. 


## 3. Getting Started
### 1) Set model configuration
Most of the configuration is same with [Edge-connect](https://github.com/knazeri/edge-connect).
For more details, refer to the sample configuration included in **./checkpoints/SAMPLE/config.yml**

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


