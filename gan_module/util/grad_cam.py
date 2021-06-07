from tensorflow import keras as tf_keras
import numpy as np

BASE_DTYPE = "float32"

"""
change image to tensor shape (1, height, width, channel)
img_path: str
size: tuple or list
dtype: str or tf dtype 
"""
def get_img_array(img_path, size, dtype=BASE_DTYPE):
    # `img` is a PIL image of size 299x299
    img = tf_keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf_keras.preprocessing.image.img_to_array(img, dtype=dtype)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


