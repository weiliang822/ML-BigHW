# 同济大学机器学习大作业

### Lab1——回归模型构建

使用的数据集为来自UCI 机器学习数据集的“Abalone“ https://archive.ics.uci.edu/dataset/1/abalone

实现数据预处理，手写最小二乘法、梯度下降法线性回归，做了Adam优化、局部加权线性回归、遗传算法等优化。



### Lab2——分类模型构建

#### 数据集

使用了两个数据集进行分类实验。一个为文本数据集，一个为图像数据集。
① 文本数据集BuddyMove
这个数据集来自UCI 机器学习数据集。这个数据集是由截至2014 年10 月发布在holidayiq.com 上的249 位评论者的目的地评论填充而成的。考虑了涵盖南印度各个目的地的6 个类别中的评论，并捕获了每位评论者（旅行者）在每个类别中的评论数量。
② 图像数据集Sort_1000pics
Sort_1000pics
数据集包含了1000 张图片，总共分为10 大类，分别是人（第0 类）、沙滩（第1 类）、建筑（第2 类）、大卡车（第3 类）、恐龙（第4 类）、大象（第5 类）、花朵（第6 类）、马（第7 类）、山峰（第8 类）和食品（第9 类）。

数据集分别来自[BuddyMove Data Set - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/476/buddymove+data+set)
和[xiao1515/Sort_1000pics: 圖像分類code (github.com)](https://github.com/xiao1515/Sort_1000pics)

#### 模型

采用了SVM 、随机森林、gbdt 、决策树还、两种贝叶斯分类等**传统机器学习算法**以及**深度学习resnet模型**进行分类实验。其中两种贝叶斯算法为手写算法，分别为高斯朴素贝叶斯和多项式朴素贝叶斯。

