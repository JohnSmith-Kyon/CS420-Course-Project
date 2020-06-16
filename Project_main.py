from UNet import *
import os
import glob
import cv2
import numpy as np
import pandas as pd
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt

from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from skimage import img_as_ubyte

# Read the images and output the numpy arrays
def get_data(image_path, label_path):
    X_train = []
    Y_train = []

    images = glob.glob(image_path + "/*.png")   # Remember to change the path.
    images.sort()
    for myImg in images:
        image = cv2.imread(myImg)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = np.array(image, dtype = 'float32') / 255.0
        X_train.append(image)

    labels = glob.glob(label_path + "/*.png")   # Remember to change the path.
    labels.sort()
    for myLabel in labels:
        label = cv2.imread(myLabel)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        label = np.array(label, dtype = 'float32') / 255.0
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    Y_train[Y_train > 0.5] = 1      # white
    Y_train[Y_train < 0.5] = 0      # black

    X_train = X_train.reshape((-1,512,512,1))
    Y_train = Y_train.reshape((-1,512,512,1))

    return X_train, Y_train

# Generate iterator for test images.
def Test_Generator(path):
    for i in range(5):  # The number of test image is 5
        image = io.imread(os.path.join(path, '%d.png' % i), as_gray=True)   # Remember to change the path.

        image = image / 255.0
        image = np.reshape(image,image.shape+(1,))
        image = np.reshape(image,(1,)+image.shape)  # Extending as same dimention as input of traing which is [2,512,512]

        yield image


def saveResult(path, npyfile, num_class = 2):
    for i, item in enumerate(npyfile):
        image = item[:,:,0]
        io.imsave(os.path.join(path,"%d_predict.png"%i),img_as_ubyte(image))


if __name__ == '__main__':
    # using GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # If something wrong happened when using GPU, the folloing config may help.
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    #
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    #     except RuntimeError as e:
    #         print(e)

    # Remember to change the path when training.
    image_path = 'new_data/new_train_set/train_img'
    label_path = 'new_data/new_train_set/train_label'
    test_path = 'new_data/new_test_set/test_img'


    train_image, train_label = get_data(image_path, label_path)

    model = unet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor = 'val_acc', verbose = 1, save_best_only = True)

    history = model.fit(train_image,
                        train_label,
                        batch_size = 2,
                        validation_split = 0.01,
                        epochs = 50,
                        callbacks = [model_checkpoint])


    test_generator = Test_Generator(test_path)
    results = model.predict_generator(test_generator, steps=5, verbose = 1) # steps = 5
    saveResult('new_data/result', results)
    model.save('new_data/result/model.h5')
