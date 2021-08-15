# Handwriting-distinguishing-via-neural-network  
@XiaoTianFan 2021

CONTENT
-----
1. BRIEF
2. REQUIREMENTs
3. DEFAULT DATA SETs
4. DEVELOPMENT LOG  

BRIEF  
------
This is a project that generated from https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork  
This project provides more functions in operating the handwriting distinguishing neural newtwork.  
In order to make it more handy, feasible, convenient, efficient, and instructional.  
Hope it will be helpful XD~

For the time being,  
you can create/train/test/backquery/save/reload your own neural network to distinguishing images;  
you can load defaulted data sets* and apply them in your use.  

REQUIREMENTs
------

All of the mentioned functions are fulfilled via Python and depend on multiple Python libraries.  

In order to run this project, ***the following supports are necessitated***:  

1.Python 3.6.X/3.7.X/3.8.X/3.9.X/...  

2.NVIDIA GPU with late released drive  

----the performance can fluctuate widely depending on the GPU model you are using  

3.NVIDIA CUDA Toolkit  

----make sure your CUDA and GPU drive is compatible in version

4.Python libraries  

----numpy  

----scipy  version: any version later then 1.3.0

----matplotlib

to use GPU version, this lib is also required  

----cupy
   
***If NVIDIA GPU and CUDA are temporarily unavailable to you, please use CPU-only version.***


DEFAULT DATA SETs
-----

Two data sets are provided in this project now:

**MNIST**  
origin: http://yann.lecun.com/exdb/mnist/

the version provided in this project is cloned from:
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/tree/master/mnist_dataset

to get full version of MNIST:  
https://pjreddie.com/projects/mnist-in-csv/

Introduction(copied from http://yann.lecun.com/exdb/mnist/):  
"The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image."

**CC10**  
A data set created by ME(qwq)

The training set includes 150 samples in total, 
consists of ten Chinese characters, that are widely used in Chinese calligraphy enlightenment, in 15 kinds of fonts.

Similarly, the testing set includes 20 samples in total.

Charaters: 人 心 永 天 火 寸 古 工 口 女


DEVELOPMENT LOG
-----

**UPCOMING**

Additional Functions:

changeable hidden layer num - different network structure will be allowed

changeable input layer num - other size of images will be allowed  

create personal data set  

Convenience improvement:

----GUI

----display all saves' name when reloading previous newworks
Other:

----expand CC10 into CC_?_

----coloured image compat
  
**v0.3**

Additional functions:

----terminal interaction system

----GPU computing

--------via NVIDIA CUDA & cupy

Debug:

----back activate image saving occasional failure

**v0.2**

Additional functions:

----network saving & reloading

----back activate image saving

Comments added

Code splited

**v0.1**

The original network from (c) Tariq Rashid, 2016
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork



