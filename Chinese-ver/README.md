# Handwriting-distinguishing-via-neural-network
@XiaoTianFan 2021  
[这里](../README.md) 是英文版文档  
[here](../README.md) is the English ver  


内容
-----
1. 概要
2. 环境需求
3. 默认数据集
4. 开发日志

概要
-----
[这里](../README.md) 是英文版文档  
[here](../README.md) is the English ver  

此项目基于 https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork 一书改写。  
此项目相较于原始程序提供了更多方便、易用、高效的功能，  
希望能以此向更多人展示神经网络的基本色彩。  
望能为诸位提供帮助~  

当前版本下可以：
创建、训练、输出反向激活图像、保存、载入您的单层神经网络  
您可以载入默认数据集* 或创建您自己的数据集进行训练(包括但不限于手写数字，汉字及其他经过标注的图像类型)

环境需求  
-----
所有以上功能均通过Python及相关Python libraries实现。  

为了正常运行本程序，***以下环境为必需：***  

1. Python 3.6.X/3.7.X/3.8.X/3.9.X/... 及以上版本  

2. NVIDIA CUDA Toolkit  

----推荐(开发)版本：CUDA 11.4/NVIDIA 显卡驱动 471+，两者需兼容  

3. 如下Python 库：  

----numpy  

----scipy (1.3.0 以上版本)  

----matplotlib  

----imageio  

以下为 GPU版本(Standard) 运行所必须Python 库

----cupy

---------注意：cupy需与CUDA版本相符，如CUDA 11.4版本环境下，使用cupy-cuda114版本  

***如果您的硬件环境内无NVIDIA GPU或CUDA暂不可用，可选择CPU-only版本***

默认数据集
-----
当前版本提供两种默认数据集

**MNIST**  
原始数据集：http://yann.lecun.com/exdb/mnist/  

本项目内提供的版本来自于：  
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/tree/master/mnist_dataset  

由于Github单文件大小限制(<25MB), 完整 mnist.csv 数据集可从此下载：  
https://pjreddie.com/projects/mnist-in-csv/

MNIST 数据集介绍(译自 http://yann.lecun.com/exdb/mnist/)：  
从本页可获得的MNIST手写数字数据库————拥有60,000个示例的训练集和10,000个示例的测试集。它是NIST提供的更大集合的子集。数字已经被尺寸标准化，并集中在一个固定大小(28 * 28)的图像中。

**CC10**
一个由项目作者创建的不成熟汉字数据集

训练集共包含十个汉字中常用且结构清晰的“字”，每“字”数据都包括15种字体。共150组数据。 
测试集汉字同上，共20组数据。

十汉字：人 心 永 天 火 寸 古 工 口 女  

开发日志
-----
***即将(keneng)推出***
新功能：  

----可自定义隐藏层数量  

便捷性改进：

----GUI界面

----更详尽可读的(双语?)注释

----交互系统改进

----提示存档名

其他：

----扩展CC10数据集

----兼容彩色图像

**v0.3**

新功能：  

----交互式系统

----GPU 加速

----可创建自定义数据集

----可自定义输入&输出层节点数

故障修复：

----反向激活图像间歇性保存失败

**v0.2**

新功能：  

----神经网络保存&重载

----反向激活图像输出&保存

添加注释

重构代码

**v0.1**

来自 (c) Tariq Rashid, 2016 的原始程序  
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork


