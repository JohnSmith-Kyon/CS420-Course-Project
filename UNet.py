import os
import numpy as np
import skimage.io as io
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model
from keras import backend as keras

## unet
def contraction(input, features):
    conv = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    pool = MaxPooling2D(pool_size = (2, 2))(conv)
    return conv, pool

def expansion(input, channel, features):
    up = Conv2D(features, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2, 2))(input))
    merge = Concatenate()([channel,up])
    conv = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
    conv = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    return conv

def bottle_neck(input, features):
    conv = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    drop = Dropout(0.2)(conv) # try the different value (0.5, 0.8)
    return drop

def unet(pretrined_weights = None, input_size = (512,512,1)):
    inputs = Input(input_size)

    ### U-Net model with three downsampling
    features = [128, 256, 512, 1024]
    pool0 = inputs
    conv1, pool1 = contraction(pool0, features[0])
    conv2, pool2 = contraction(pool1, features[1])
    conv3, pool3 = contraction(pool2, features[2])

    drop = bottle_neck(pool3, features[3])

    up1 = expansion(drop, conv3, features[2])
    up2 = expansion(up1, conv2, features[1])
    up3 = expansion(up2, conv1, features[0])

    outputs = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(up3)

    ### U-Net with four times downsampling
    #features = [64, 128, 256, 512, 1024]
    #pool0 = inputs
    #conv1, pool1 = contraction(pool0, features[0])
    #conv2, pool2 = contraction(pool1, features[1])
    #conv3, pool3 = contraction(pool2, features[2])
    #conv4, pool4 = contraction(pool3, features[3])

    #drop = bottle_neck(pool4, features[4])

    #up1 = expansion(drop, conv4, features[3])
    #up2 = expansion(up1, conv3, features[2])
    #up3 = expansion(up2, conv2, features[1])
    #up4 = expansion(up3, conv1, features[0])
    #outputs = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(up4)


    ### U-Net with five times downsampling
    #features = [64, 128, 256, 512, 1024, 2048]
    #pool0 = inputs
    #conv1, pool1 = contraction(pool0, features[0])
    #conv2, pool2 = contraction(pool1, features[1])
    #conv3, pool3 = contraction(pool2, features[2])
    #conv4, pool4 = contraction(pool3, features[3])
    #conv5, pool5 = contraction(pool4, features[4])

    #drop = bottle_neck(pool5, features[5])

    #up1 = expansion(drop, conv5, features[4])
    #up2 = expansion(up1, conv4, features[3])
    #up3 = expansion(up2, conv3, features[2])
    #up4 = expansion(up3, conv2, features[1])
    #up5 = expansion(up4, conv1, features[0])
    #outputs = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(up5)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr = 3e-4), loss="binary_crossentropy", metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='model.jpg') # the image of model processing
    return model
