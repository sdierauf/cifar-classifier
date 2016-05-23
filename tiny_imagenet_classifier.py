'''Trains a simple convnet on the tiny imagenet dataset

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# System
from __future__ import print_function
import numpy as np
import os
from PIL import Image
np.random.seed(1337)  # for reproducibility

#Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.datasets import cifar10
from keras.regularizers import WeightRegularizer, ActivityRegularizer 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from plotter import Plotter

#Custom
from val_load import get_annotations_map

loss_functions = ['categorical_crossentropy', 'hinge', 'squared_hinge']
#loss_functions = ['hinge']
# num_classes_arr = [2, 5, 8]
# num_classes_arr = [10, 100, 200]
# num_classes_arr = [10]
for loss_function in loss_functions:
    # for num_classes in num_classes_arr: # num classes loop
    num_classes = 10

    print()
    print()
    print('===========================')
    print('Testing: ' + loss_function + ' with ' + str(num_classes) + ' classes')
    print('===========================')
    print()

    batch_size = 32
    nb_classes = 10
    nb_epoch = 200
    data_augmentation = True

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

        # batch_size = 64
        # nb_classes = 200
        # nb_epoch = 100
        # classes_to_load = num_classes
        # nb_classes = min(classes_to_load, nb_classes)

        # print('Training on ' + str(classes_to_load) + ' classes')

        # X_train=np.zeros([classes_to_load*500,3,64,64],dtype='uint8')
        # y_train=np.zeros([classes_to_load*500], dtype='uint8') #TODO See if works like this

        # trainPath='./tiny-imagenet-200/train'

        # print('loading training images...');
        # # would be nice to show a progress bar here..
        # i=0
        # j=0
        # tempIndex2imagenetClass={}
        # imagenetClass2TempIndex={} #inverse mapping
        # for sChild in os.listdir(trainPath):
        #     sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        #     tempIndex2imagenetClass[j]=sChild
        #     imagenetClass2TempIndex[sChild] = j
        #     for c in os.listdir(sChildPath):
        #         X=np.array(Image.open(os.path.join(sChildPath,c)))
        #         if len(np.shape(X))==2:
        #             X_train[i]=np.array([X,X,X])
        #         else:
        #             # print(X.shape)
        #             # p = 0
        #             for d in range(3):
        #                 for w in range(64):
        #                     for h in range(64):
        #                         X_train[i][d][w][h] = X[w][h][d]
        #         y_train[i]=j
        #         i+=1
        #     # print('finished loading ' + str(j) + ' out of ' + str(classes_to_load) + ' classes')
        #     j+=1
        #     if (j >= classes_to_load):
        #         break

        # print('finished loading training images')

        # val_annotations_map = get_annotations_map()

        # X_test = np.zeros([classes_to_load*500,3,64,64],dtype='uint8')
        # y_test = np.zeros([classes_to_load*500], dtype='uint8')


        # print('loading test images...')

        # i = 0
        # testPath='./tiny-imagenet-200/val/images'
        # for sChild in os.listdir(testPath):
        #     if val_annotations_map[sChild] in imagenetClass2TempIndex:
        #         # print(val_annotations_map[sChild] + ' in imagenetClass2TempIndex')
        #         sChildPath = os.path.join(testPath, sChild)
        #         # print(sChildPath)
        #         X=np.array(Image.open(sChildPath))
        #         if len(np.shape(X))==2:
        #             X_test[i]=np.array([X,X,X])
        #         else:
        #             for d in range(3):
        #                 for w in range(64):
        #                     for h in range(64):
        #                         X_test[i][d][w][h] = X[w][h][d]

        #         y_test[i]=imagenetClass2TempIndex[val_annotations_map[sChild]]
        #         i+=1
        #     else:
        #         pass
        #         # print(val_annotations_map[sChild] + ' not in imagenetClass2TempIndex')


        # print('finished loading test images')

        # # input image dimensions
        # img_rows, img_cols = 64, 64
        # # number of convolutional filters to use
        # nb_filters = 32
        # # size of pooling area for max pooling
        # nb_pool = 2
        # # convolution kernel size
        # nb_conv = 3
        #


    #the data, shuffled and split between train and test sets
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')
    # # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)

        # generate flips etc of the images

        # datagen.fit(X_train)

    model = Sequential()

    model.add(Convolution2D(96, 3, 3, border_mode='same', input_shape=(3, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
   
    model.add(Convolution2D(96, 3, 3, border_mode='same', subsample=(2,2)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(192, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(2,2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(192, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 1, 1, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 1, 1, border_mode='same'))
    model.add(Activation('relu'))
    AveragePooling2D(pool_size=(6, 6), strides=None, border_mode='same', dim_ordering='th')
    # model.add(Activation('relu'))
    model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    model.add(Dense(10))#pretrained weights assume only 100 outputs, we need to train this layer from scratch

    # model.add(AveragePooling2D(pool_size=(6, 6), strides=None, border_mode='valid', dim_ordering='th'))
    model.add(Activation('softmax'))


# ======================
        # model.add(Convolution2D(64, 3, 3, border_mode='same',
        #                         input_shape=(3, img_rows, img_cols)))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Convolution2D(64, 3, 3, border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        # model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(nb_classes))
        # model.add(Activation('softmax'))
# ======================


    # #conv-spatial batch norm - relu #1 
    # model.add(ZeroPadding2D((2,2),input_shape=(3,32,32)))
    # model.add(Convolution2D(64,5,5,subsample=(2,2),W_regularizer=WeightRegularizer(l1=1e-7,l2=1e-7)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 

    # #conv-spatial batch norm - relu #2
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(64,3,3,subsample=(1,1)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 

    # #conv-spatial batch norm - relu #3
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128,3,3,subsample=(2,2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 
    # model.add(Dropout(0.25)) 

    # #conv-spatial batch norm - relu #4
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128,3,3,subsample=(1,1)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 

    # #conv-spatial batch norm - relu #5
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256,3,3,subsample=(2,2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 

    # #conv-spatial batch norm - relu #6
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256,3,3,subsample=(1,1)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 
    # model.add(Dropout(0.25))

    # #conv-spatial batch norm - relu #7
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512,3,3,subsample=(2,2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 

    # #conv-spatial batch norm - relu #8
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512,3,3,subsample=(1,1)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 
    

    # #conv-spatial batch norm - relu #9
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(1024,3,3,subsample=(2,2)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25)) 

    # #Affine-spatial batch norm -relu #10 
    # model.add(Flatten())
    # model.add(Dense(512,W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))
    # model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    # model.add(Activation('relu')) 
    # model.add(Dropout(0.5)) 

    # #affine layer w/ softmax activation added 
    # model.add(Dense(num_classes,activation='softmax',W_regularizer=WeightRegularizer(l1=1e-5,l2=1e-5)))#pretrained weights assume only 100 outputs, we need to train this layer from scratch

    # if loss_function is 'categorical_crossentropy':
   # model.add(Activation('softmax'))
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss_function,
                  optimizer=sgd,
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    fpath = 'loss-' + loss_function + '-' + str(num_classes)
    datagen.fit(X_train)


    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                verbose=1, validation_data=(X_test, Y_test),
                callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])

    # model.fit(X_train, Y_train, batch_size=64, nb_epoch=nb_epoch,
    #           verbose=1, validation_data=(X_test, Y_test),
    #           callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])



    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

