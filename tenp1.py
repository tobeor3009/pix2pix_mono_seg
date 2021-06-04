
# base module
import threading
from queue import Queue

# external module
import cv2
import tensorflow as tf
import numpy as np
import segmentation_models as sm
from sklearn.utils import shuffle


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
    def __init__(self, image_path_list, mask_paths_list, backbone=BACKBONE):
        self.image_path_list = image_path_list
        self.mask_paths_list = mask_paths_list
        self.preprocess_input = sm.get_preprocessing(backbone)

    def __getitem__(self, i):
        image_path = self.image_path_list[i]
        mask_path = self.mask_paths_list[i]

        image = imread(image_path, channel="rgb")
        mask = imread(mask_path)

        # normalize image: [0, 255] => [-1, 1]
        # normalize mask: [0, 255] => [0, 1]
        image = (image / 127.5) - 1
        # image = self.preprocess_input(image)
        mask = (mask / 2)

        return image, mask

    def shuffle(self):
        self.image_paths, self.mask_paths = \
            shuffle(self.image_paths, self.mask_paths)


class SegDataloader(tf.keras.utils.Sequence):

    def __init__(self, image_path_list, mask_paths_list, batch_size,
                 shuffle=True, backbone=BACKBONE):
        self.data_getter = SegDataGetter(
            image_path_list, mask_paths_list, backbone)
        self.batch_size = batch_size
        self.source_data_shape = self.data_getter[0][0].shape
        self.gournd_truth_data_shape = self.data_getter[0][1].shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, i):

        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty((self.batch_size, *self.source_data_shape))
        batch_y = np.empty((self.batch_size, *self.gournd_truth_data_shape))
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


class DataGetter:

    def __init__(self, image_paths, label_dict=None):
        self.image_paths = image_paths
        self.label_dict = {"tumor": 1, "non_tumor": 0}
        self.num_classes = len(self.label_dict)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image_dir_name = get_parent_dir_name(image_path)

        image = imread(image_path, channel="rgb")

        label = self.label_dict[image_dir_name]
        label = tf.keras.utils.to_categorical(
            label, num_classes=self.num_classes)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def shuffle(self):
        np.random.shuffle(self.image_paths)


class CacheDataloader(tf.keras.utils.Sequence):

    def __init__(self, data_getter, data_shape, batch_size, shuffle=True):
        self.data_getter = data_getter
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.batch_num = len(self.data_getter) // self.batch_size

        self.cache_event = threading.Event()
        self.cache_queue = Queue()
        self.cache_index = None
        self.cache_thread = threading.Thread(
            target=self.fill_queue, daemon=True).start()

        self.on_epoch_end()

    def __getitem__(self, i):

        if i < self.batch_num - 1:
            if i == self.cache_index:
                batch = self.get_and_set_cache(i)
            elif self.cache_index is None:
                self.set_cache(i + 1)
                batch = self.get_item_from_getter(i)
            elif i != self.cache_index:
                # clear unused cache and put next cache in queue
                _ = self.get_and_set_cache(i)
                batch = self.get_item_from_getter(i)
        else:
            self.cache_index = None
            batch = self.get_item_from_getter(i)

        return batch

    def get_item_from_getter(self, i):
        start = i * self.batch_size
        end = start + self.batch_size

        batch_x = np.empty((self.batch_size, *self.data_shape))
        batch_y = []
        for batch_index, total_index in enumerate(range(start, end)):
            data = self.data_getter[total_index]
            batch_x[batch_index] = (data[0] / 127.5) - 1
            batch_y.append(data[1])
        batch_y = np.array(batch_y)

        return batch_x, batch_y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.batch_num

    def on_epoch_end(self):
        if self.shuffle:
            self.data_getter.shuffle()

    def set_cache(self, i):
        self.cache_index = i
        self.cache_event.set()

    def get_cache(self):
        batch = self.cache_queue.get()
        return batch

    def get_and_set_cache(self, i):
        batch = self.get_cache()
        self.set_cache(i + 1)
        return batch

    def fill_queue(self):
        while True:
            # wait until set_cache's cache_event.set()
            self.cache_event.wait()
            batch = self.get_item_from_getter(self.cache_index)
            self.cache_queue.put(batch)
            self.cache_event.clear()
