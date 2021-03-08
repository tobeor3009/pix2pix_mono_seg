import cv2
import numpy as np

import os
import random
from glob import glob
from collections import deque


TEST_DATA_NUM = 20


class DataLoader:

    def __init__(self, dataset_name, img_res=(512, 512), on_memory=False, test=False):

        self.img_res = img_res
        self.on_memory = on_memory
        self.train_loaded_data = None
        self.valid_loaded_data = None
        # -----------------
        #  get FilePath
        # -----------------

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

        self.train_image_file_paths = np.array(
            glob(train_image_data_path_regexp))
        self.train_mask_file_paths = np.array(
            glob(trian_mask_data_path_regexp))
        self.valid_image_file_paths = np.array(
            glob(valid_image_data_path_regexp))
        self.valid_mask_file_paths = np.array(
            glob(valid_mask_data_path_regexp))

        if test:
            self.train_image_file_paths = self.train_image_file_paths[:TEST_DATA_NUM]
            self.train_mask_file_paths = self.train_mask_file_paths[:TEST_DATA_NUM]
            self.valid_image_file_paths = self.valid_image_file_paths[:TEST_DATA_NUM]
            self.valid_mask_file_paths = self.valid_mask_file_paths[:TEST_DATA_NUM]

        self.train_data_length = len(self.train_image_file_paths)
        self.valid_data_length = len(self.valid_image_file_paths)

        self.train_data_index = np.arange(self.train_data_length)
        self.valid_data_index = np.arange(self.valid_data_length)

        # -----------------
        #  load File if on_memory
        # -----------------

        if self.on_memory:
            self.train_loaded_data, self.valid_loaded_data = self._load_whole_data()

        # -----------------
        #  check mask and image num equal
        # -----------------

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

    def get_data(self, data_mode, index=None):

        if self.on_memory:
            if data_mode == "train":
                if index is None:
                    data_tuple = self.train_loaded_data
                else:
                    data_tuple = (
                        self.train_loaded_data[0][index], self.train_loaded_data[1][index])
            if data_mode == "valid":
                if index is None:
                    data_tuple = self.valid_loaded_data
                else:
                    data_tuple = (
                        self.valid_loaded_data[0][index], self.valid_loaded_data[1][index])
        else:
            data_tuple = self.__get_processed_imgs(
                data_mode=data_mode, index=index)

        return data_tuple

    def _load_whole_data(self):
        tuple_train_data = self.__get_processed_imgs(data_mode="train")
        tuple_valid_data = self.__get_processed_imgs(data_mode="valid")
        return tuple_train_data, tuple_valid_data

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

    def __get_processed_imgs(self, data_mode, index=None):

        image_list, mask_list = deque(), deque()

        if data_mode == "train":
            image_paths = self.train_image_file_paths
            mask_paths = self.train_mask_file_paths
        if data_mode == "valid":
            image_paths = self.valid_image_file_paths
            mask_paths = self.valid_mask_file_paths

        if index is None:
            pass
        else:
            image_paths = image_paths[index]
            mask_paths = mask_paths[index]

        for image_path, mask_path in zip(image_paths, mask_paths):
            image = self.__imread(image_path, channel='bgr')
            mask = self.__imread(mask_path)

            image_list.append(np.array(image))
            mask_list.append(np.array(mask))

        image_list = np.array(image_list, dtype=np.float32) / 127.5 - 1.0
        mask_list = np.round(np.array(mask_list, dtype=np.float32) / 255)

        return image_list, mask_list
