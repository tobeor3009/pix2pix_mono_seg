# base module

# external module
import tensorflow as tf
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
                 image_path_list=None,
                 label_to_index_dict=None):
        self.image_path_list = image_path_list
        self.label_to_index_dict = label_to_index_dict
        self.num_classes = len(self.label_to_index_dict)

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        image_dir_name = get_parent_dir_name(image_path)

        image = imread(image_path, channel="rgb")
        image = (image / 127.5) - 1

        label = self.label_to_index_dict[image_dir_name]
        label = tf.keras.utils.to_categorical(
            label, num_classes=self.num_classes)
        return image, label

    def __len__(self):
        return len(self.image_path_list)

    def shuffle(self):
        syncron_shuffle(self.image_path_list)


class ClassifyDataloader(tf.keras.utils.Sequence):

    def __init__(self,
                 image_path_list=None,
                 label_to_index_dict=None,
                 batch_size=None,
                 shuffle=True):
        self.data_getter = ClassifyDataGetter(image_path_list=image_path_list,
                                              label_to_index_dict=label_to_index_dict,
                                              )
        self.batch_size = batch_size
        self.num_classes = len(label_to_index_dict)
        self.source_data_shape = self.data_getter[0][0].shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty((self.batch_size, *self.source_data_shape))
        batch_y = np.empty((self.batch_size, self.num_classes))
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
