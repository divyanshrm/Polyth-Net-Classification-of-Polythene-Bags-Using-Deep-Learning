# Polyth-Net-Classification-of-Polythene-Bags-Using-Deep-Learning
Backend work for the research publication titled - Polyth-Net: Classification of Polythene Bags Using Deep Dearning

This project aims to classify polythene bags efficiently from other garbage materials to abolish and minmize the human labour
as this job is very hazardous to the healths of humans working on these segregation tasks
The classes are as follows

0: Non-plastic

1: Plastic but not Polythene

2: Polythene bags

The model uses an Xception network combined with data augmentation from keras to produce good results on test data.


Link to the research publication- https://arxiv.org/abs/2008.07592

Dependencies: 

Tensorflow

Keras

OpenCV 

Python 3.7

Guide to train the model the model:

modify the dataset path.

python train.py



The Model will be saved as an .h5 file.

Abstract: 

Polythene has always been a threat to the environment since its invention. It is non-biodegradable and very difficult to recycle. Even after many awareness campaigns and practices, Separation of polythene bags from waste has been a challenge for human civilization. The primary method of segregation deployed is manual handpicking, which causes a dangerous health hazards to the workers and is also highly inefficient due to human errors. In this paper I have designed and researched on image-based classification of polythene bags using a deep-learning model and its efficiency. This paper focuses on the architecture and statistical analysis of its performance on the data set as well as problems experienced in the classification. It also suggests a modified loss function to specifically detect polythene irrespective of its individual features. It aims to help the current environment protection endeavours and save countless lives lost to the hazards caused by current methods.
