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
