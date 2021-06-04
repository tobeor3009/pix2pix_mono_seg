# base module

# external module
import cv2
import tensorflow as tf
import numpy as np
import segmentation_models as sm
from sklearn.utils import shuffle as syncron_shuffle


def imread(path, channel=None):
    image_byte_stream = open(path.encode("utf-8"), "rb")
    image_byte_array = bytearray(image_byte_stream.read())
    image_numpy_array = np.asarray(image_byte_array, dtype=np.uint8)
    image_numpy_array = cv2.imdecode(
        image_numpy_array, cv2.IMREAD_UNCHANGED)
    if channel == "rgb":
        image_numpy_array = cv2.cvtColor(
            image_numpy_array, cv2.COLOR_BGR2RGB)
    if len(image_numpy_array.shape) == 2:
        image_numpy_array = np.expand_dims(image_numpy_array, axis=-1)
    return image_numpy_array


def get_parent_dir_name(path):
    return path.split('/')[-2]


"""
Expect Data Path Structure

train - image
      - mask
valid - image
      - mask
test  - image
      - mask
"""

BACKBONE = "resnet34"


def imread(path, channel=None):
    image_byte_stream = open(path.encode("utf-8"), "rb")
    image_byte_array = bytearray(image_byte_stream.read())
    image_numpy_array = np.asarray(image_byte_array, dtype=np.uint8)
    image_numpy_array = cv2.imdecode(
        image_numpy_array, cv2.IMREAD_UNCHANGED)
    if channel == "rgb":
        image_numpy_array = cv2.cvtColor(
            image_numpy_array, cv2.COLOR_BGR2RGB)
    if len(image_numpy_array.shape) == 2:
        image_numpy_array = np.expand_dims(image_numpy_array, axis=-1)
    return image_numpy_array


def get_parent_dir_name(path):
    return path.split('/')[-2]


"""
Expect Data Path Structure

train - image
      - mask
valid - image
      - mask
test  - image
      - mask
"""

BACKBONE = "resnet34"


class SegDataGetter():
    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 backbone=BACKBONE):
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        assert len(image_path_list) == len(mask_path_list), \
            f"image_num = f{len(image_path_list)}, mask_num = f{len(mask_path_list)}"
        self.preprocess_input = sm.get_preprocessing(backbone)

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        mask_path = self.mask_path_list[i]

        image = imread(image_path, channel="rgb")
        mask = imread(mask_path)

        # normalize image: [0, 255] => [-1, 1]
        # normalize mask: [0, 255] => [0, 1]
        image = (image / 127.5) - 1
        image = self.preprocess_input(image)
        mask = (mask / 255)

        return image, mask

    def __len__(self):
        return len(self.image_path_list)

    def shuffle(self):
        self.image_path_list, self.mask_path_list = \
            syncron_shuffle(self.image_path_list, self.mask_path_list)


class SegDataloader(tf.keras.utils.Sequence):

    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 batch_size=4,
                 shuffle=True,
                 backbone=BACKBONE,
                 dtype="float32"):
        self.data_getter = SegDataGetter(
            image_path_list, mask_path_list, backbone)
        self.batch_size = batch_size
        self.dtype = dtype
        self.image_data_shape = self.data_getter[0][0].shape
        self.mask_data_shape = self.data_getter[0][1].shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        batch_y = np.empty(
            (self.batch_size, *self.mask_data_shape), dtype=self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            data = self.data_getter[total_index]
            batch_x[batch_index] = data[0]
            batch_y[batch_index] = data[1]
        batch_y = np.array(batch_y)

        return batch_x, batch_y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()


class SegSingleDataGetter():
    def __init__(self,
                 image_path_list=None,
                 preprocess=False,
                 backbone=BACKBONE):
        self.image_path_list = image_path_list
        self.preprocess = preprocess
        self.preprocess_input = sm.get_preprocessing(backbone)

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        if self.preprocess:
            image = imread(image_path, channel="rgb")
            image = (image / 127.5) - 1
            image = self.preprocess_input(image)
        else:
            image = imread(image_path)
            image = (image / 2)

        return image

    def __len__(self):
        return len(self.image_path_list)

    def shuffle(self):
        self.image_path_list = syncron_shuffle(self.image_path_list)


class SegSingleDataloader(tf.keras.utils.Sequence):

    def __init__(self,
                 image_path_list=None,
                 batch_size=4,
                 shuffle=True,
                 preprocess=False,
                 backbone=BACKBONE,
                 dtype="float32"):
        self.data_getter = SegSingleDataGetter(
            image_path_list=image_path_list,
            preprocess=preprocess,
            backbone=backbone)
        self.batch_size = batch_size
        self.dtype = dtype
        self.image_data_shape = self.data_getter[0].shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty(
            (self.batch_size, *self.image_data_shape), dtype=self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            data = self.data_getter[total_index]
            batch_x[batch_index] = data[0]

        return batch_x

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()
