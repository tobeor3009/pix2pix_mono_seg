import cv2
import numpy as np

import os
import random
from glob import glob
from collections import deque


class DataLoader:
    def __init__(self, dataset_name, img_res=(512, 512), on_memory=False, test=False, data_mode=None):

        self.img_res = img_res
        self.on_memory = on_memory
        self.data_mode = data_mode
        
        current_path = os.getcwd()
        dataset_path = os.path.join(current_path, "datasets", dataset_name)
        train_image_data_path_regexp = os.path.join(
            dataset_path, "train", "image", "*")
        trian_mask_data_path_regexp = os.path.join(
            dataset_path, "train", "mask", "*")
        valid_image_data_path_regexp = os.path.join(
            dataset_path, "valid", "image", "*")
        valid_mask_data_path_regexp = os.path.join(
            dataset_path, "valid", "mask", "*")

        self.train_image_file_paths = glob(train_image_data_path_regexp)
        self.train_mask_file_paths = glob(trian_mask_data_path_regexp)
        self.valid_image_file_paths = glob(valid_image_data_path_regexp)
        self.valid_mask_file_paths = glob(valid_mask_data_path_regexp)
        
        if test:
            self.train_image_file_paths = self.train_image_file_paths[:20]
            self.train_mask_file_paths = self.train_mask_file_paths[:20]
            self.valid_image_file_paths = self.valid_image_file_paths[:20]
            self.valid_mask_file_paths = self.valid_mask_file_paths[:20]

        self.train_data_length = len(self.train_image_file_paths)
        self.valid_data_length = len(self.valid_image_file_paths)
            
        self.train_data_index = np.arange(self.train_data_length)
        self.valid_data_index = np.arange(self.valid_data_length)
        
        if not len(self.train_mask_file_paths) == len(self.train_mask_file_paths):
            raise Exception(
                f"the amount of train image and mask must be the same. \
                    image : {len(self.train_image_file_paths)}, \
                    mask : {len(self.train_mask_file_paths)} ")

        if not len(self.valid_image_file_paths) == len(self.valid_mask_file_paths):
            raise Exception(
                f"the amount of vaild image and mask must be the same. \
                    image : {len(self.valid_image_file_paths)}, \
                    mask : {len(self.valid_mask_file_paths)} ")

    def load_data(self, batch_size=1):

        train_data_random_index = np.random.choice(
            self.train_data_index, size=batch_size)

        valid_data_random_index = np.random.choice(
            self.valid_data_index, size=batch_size)

        train_image_batch_paths = [self.train_image_file_paths[index]
                                   for index in train_data_random_index]
        train_mask_batch_paths = [self.train_mask_file_paths[index]
                                  for index in train_data_random_index]

        valid_image_batch_paths = [self.valid_image_file_paths[index]
                                   for index in valid_data_random_index]
        valid_mask_batch_paths = [self.valid_mask_file_paths[index]
                                  for index in valid_data_random_index]

        tuple_train_data = self.__get_processed_imgs_array(
            train_image_batch_paths, train_mask_batch_paths)

        tuple_valid_data = self.__get_processed_imgs_array(
            valid_image_batch_paths, valid_mask_batch_paths)

        return tuple_train_data, tuple_valid_data

    def load_all(self):

        if self.on_memory:
                tuple_train_data = self.__get_processed_imgs_array(
                    self.train_image_file_paths, self.train_mask_file_paths)
                tuple_valid_data = self.__get_processed_imgs_array(
                    self.valid_image_file_paths, self.valid_mask_file_paths)
                return tuple_train_data, tuple_valid_data

        else:
            # TBD : add get vaild data
            if random.random() > 0.5:
                random.shuffle(self.train_image_file_paths)
                random.shuffle(self.train_mask_file_paths)
            return self.__get_processed_imgs_iter(self.train_image_file_paths, self.train_mask_file_paths)

    def __imread(self, path, channel=None):
        image_byte_stream = open(path.encode("utf-8"), "rb")
        image_byte_array = bytearray(image_byte_stream.read())
        image_numpy_array = np.asarray(image_byte_array, dtype=np.uint8)
        image_numpy_array = cv2.imdecode(
            image_numpy_array, cv2.IMREAD_UNCHANGED)

        if channel == 'bgr':
            image_numpy_array = cv2.cvtColor(
                image_numpy_array, cv2.COLOR_BGR2RGB)

        return image_numpy_array

    # separate imgs and make png data range(-255,255) to (-1,1)

    def __get_processed_imgs_array(self, image_paths, mask_paths):

        image_list, mask_list = deque(), deque()
        for image_path, mask_path in zip(image_paths, mask_paths):
            image = self.__imread(image_path, channel='bgr')
            mask = self.__imread(mask_path)

            image_list.append(np.array(image))
            mask_list.append(np.array(mask))

        image_list = np.array(image_list, dtype=np.float32) / 127.5 - 1.0
        mask_list = np.round(np.array(mask_list, dtype=np.float32) / 255)

        return image_list, mask_list

    def __get_processed_imgs_iter(self, image_paths, mask_paths):

        for image_path, mask_path in zip(image_paths, mask_paths):
            image = self.__imread(image_path, channel='bgr')
            mask = self.__imread(mask_path)

            image = np.array(
                image, dtype=np.float32) / 127.5 - 1.0
            mask = np.round(np.array(mask, dtype=np.float32) / 255)

            yield (image, mask)
