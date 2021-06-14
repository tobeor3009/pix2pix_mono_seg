# base module

# external module
import tensorflow as tf
import cv2
import numpy as np
from sklearn.utils import shuffle as syncron_shuffle

# this library module
from .utils import imread, get_parent_dir_name

"""
Expect Data Path Structure

train - class_names
valid - class_names
test - class_names

Example)
train - non_tumor
      - tumor
label_dict = {"non_tumor":0, "tumor":1}
"""


class ClassifyDataGetter:

    def __init__(self,
                 image_path_list,
                 label_to_index_dict,
                 preprocess_input,
                 target_size,
                 interpolation):
        self.image_path_list = image_path_list
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        image_dir_name = get_parent_dir_name(image_path)

        image = imread(image_path, channel="rgb")
        if self.interpolation == "bilinear":
            image = cv2.resize(image, self.target_size, cv2.INTER_LINEAR)

        if self.preprocess_input:
            image = self.preprocess_input(image)
        else:
            image = (image / 127.5) - 1

        label = self.label_to_index_dict[image_dir_name]
        label = tf.keras.utils.to_categorical(
            label, num_classes=self.num_classes)

        return image, label

    def __len__(self):
        return len(self.image_path_list)

    def shuffle(self):
        self.image_path_list = syncron_shuffle(self.image_path_list)

    def check_label_status(self):
        data_len = self.__len__()
        for index in range(data_len):
            image_path = self.image_path_list[index]
            image_dir_name = get_parent_dir_name(image_path)
            label = self.label_to_index_dict[image_dir_name]
            print(image_path)
            print(label)
            print(tf.keras.utils.to_categorical(
                label, num_classes=self.num_classes))


class ClassifyDataloader(tf.keras.utils.Sequence):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = ClassifyDataGetter(image_path_list=image_path_list,
                                              label_to_index_dict=label_to_index_dict,
                                              preprocess_input=preprocess_input,
                                              target_size=target_size,
                                              interpolation=interpolation
                                              )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.source_data_shape = self.data_getter[0][0].shape
        if target_size is None:
            target_size = self.source_data_shape[:2]
            self.data_getter.target_size = target_size
            self.source_data_shape[:2] = target_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty(
            (self.batch_size, *self.source_data_shape), dtype=self.dtype)
        batch_y = np.empty(
            (self.batch_size, self.num_classes), dtype=self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            data = self.data_getter[total_index]
            batch_x[batch_index] = data[0]
            batch_y[batch_index] = data[1]

        return batch_x, batch_y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()


class BinaryClassifyDataGetter:

    def __init__(self,
                 image_path_list,
                 label_to_index_dict,
                 preprocess_input,
                 target_size,
                 interpolation):
        self.image_path_list = image_path_list
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation

        assert self.num_classes == 2, f"label_to_index_dict: {label_to_index_dict}"

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        image_dir_name = get_parent_dir_name(image_path)

        image = imread(image_path, channel="rgb")
        if self.interpolation == "bilinear":
            image = cv2.resize(image, self.target_size, cv2.INTER_LINEAR)

        if self.preprocess_input:
            image = self.preprocess_input(image)
        else:
            image = (image / 127.5) - 1

        label = self.label_to_index_dict[image_dir_name]
        return image, label

    def __len__(self):
        return len(self.image_path_list)

    def shuffle(self):
        self.image_path_list = syncron_shuffle(self.image_path_list)

    def check_label_status(self):
        data_len = self.__len__()
        for index in range(data_len):
            image_path = self.image_path_list[index]
            image_dir_name = get_parent_dir_name(image_path)
            label = self.label_to_index_dict[image_dir_name]
            print(image_path)
            print(image_dir_name)
            print(label)


class BinaryClassifyDataloader(tf.keras.utils.Sequence):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 preprocess_input=None,
                 target_size=None,
                 interpolation="bilinear",
                 shuffle=True,
                 dtype="float32"):
        self.data_getter = BinaryClassifyDataGetter(image_path_list=image_path_list,
                                                    label_to_index_dict=label_to_index_dict,
                                                    preprocess_input=preprocess_input,
                                                    target_size=target_size,
                                                    interpolation=interpolation
                                                    )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.source_data_shape = self.data_getter[0][0].shape
        if target_size is None:
            target_size = self.source_data_shape[:2]
            self.data_getter.target_size = target_size
            self.source_data_shape[:2] = target_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty((self.batch_size, *self.source_data_shape), self.dtype)
        batch_y = np.empty((self.batch_size,), self.dtype)
        for batch_index, total_index in enumerate(range(start, end)):
            data = self.data_getter[total_index]
            batch_x[batch_index] = data[0]
            batch_y[batch_index] = data[1]

        return batch_x, batch_y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.data_getter) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()
