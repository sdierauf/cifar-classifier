'''Trains a simple convnet on the tiny imagenet dataset

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import os
from PIL import Image
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 128
nb_classes = 200
nb_epoch = 12

X_train=np.zeros([200*500,3,64,64],dtype='uint8')
y_train=np.zeros([200*500]) #TODO See if works like this

sPath='./tiny-imagenet-200/train'
i=0
j=0
mapping={}
for sChild in os.listdir(sPath):
    sChildPath = os.path.join(os.path.join(sPath,sChild),'images')
    mapping[j]=sChild
    for c in os.listdir(sChildPath):
        X=np.array(Image.open(os.path.join(sChildPath,c)))
        if len(np.shape(X))==2:
            X_train[i]=np.array([X,X,X])
        else:
            X_train[i]=np.transpose(X,(2,0,1))
        y_train[i]=j
        i+=1
    j+=1

# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
#nb_conv = 3
#



#the data, shuffled and split between train and test sets
X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
X_train /= 255
#X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)