import io
import os

import numpy as np
import pandas as pd

from PIL import Image
from keras.datasets import mnist

def __save_jpg_format(img):
    img_2d = img.reshape(int(np.sqrt(img.shape[0])), -1)
    img_pil = Image.fromarray(img_2d)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    img_bytes = bytearray(buffer.getvalue())
    return img_bytes

def create_mnist_bytearray_df(label_col: str = 'label', n_samples: int = 500):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    flattened_images = []
    for img in x_train.reshape(x_train.shape[0], -1):
        img_bytes = __save_jpg_format(img)
        flattened_images.append([img_bytes])
    
    return pd.DataFrame(list(zip(flattened_images, y_train)), columns=['image', label_col])[:n_samples]

def create_mnist_tuple(n_samples: int = 500):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    #y = y.reshape(-1, 1)
    #data = np.hstack((x.reshape(x.shape[0], -1), y))
    #return data[:n_samples]
    # y = y[:, np.newaxis, np.newaxis]
    # data = np.concatenate((x, y), axis=1)
    # return data[:n_samples]
    #return tupel
    return x[:n_samples], y[:n_samples]

def create_glass_df():
    path = os.path.join(os.path.dirname(__file__), '../../data/glass.csv')
    return pd.read_csv(path)

def create_m4_df(n_samples: int = 2000):
    return pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")[:n_samples]

#---------------------------------------------------------------------------------------------#
#=============================================================================================#
#---------------------------------------------------------------------------------------------#

mnist_byte_df = create_mnist_bytearray_df()
mnist_tp = create_mnist_tuple()
glass_df = create_glass_df()
m4_df = create_m4_df()