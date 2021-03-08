import cv2
import numpy as np

import os
import random
from glob import glob
from collections import deque


class DataLoader:
    def __init__(self, dataset_name, img_res=(512, 512)):

        current_path = os.getcwd()
        image_data_path_regexp = os.path.join(
            current_path, "datasets", dataset_name, "image", "*")
        mask_data_path_regexp = os.path.join(
            current_path, "datasets", dataset_name, "mask", "*")
        self.image_file_paths = glob(image_data_path_regexp)
        self.mask_file_paths = glob(mask_data_path_regexp)
        self.img_res = img_res
        self.data_length = len(self.image_file_paths)
        self.data_index = np.arange(self.data_length)
        if not len(self.image_file_paths) == len(self.mask_file_paths):
            raise Exception(
                f"the amount of image and mask must be the same. \
                    image : {len(self.image_file_paths)}, \
                    mask : {len(self.mask_file_paths)} ")

    def load_data(self, batch_size=1):

        data_random_index = np.random.choice(
            self.data_index, size=batch_size)

        image_batch_paths = [self.image_file_paths[index]
                             for index in data_random_index]
        mask_batch_paths = [self.mask_file_paths[index]
                            for index in data_random_index]

        return self.__get_processed_imgs_array(image_batch_paths, mask_batch_paths)

    def load_all(self, on_memory=False, test=False):

        if on_memory:
            if test:
                return self.__get_processed_imgs_array(self.image_file_paths[:20], self.mask_file_paths[:20])
            else :
                return self.__get_processed_imgs_array(self.image_file_paths, self.mask_file_paths)
        
        else:
            if random.random() > 0.5:
                random.shuffle(self.image_file_paths)
                random.shuffle(self.mask_file_paths)
            return self.__get_processed_imgs_iter(self.image_file_paths, self.mask_file_paths)

    def __imread(self, path, channel=None):
        image_byte_stream = open(path.encode("utf-8"), "rb")
        image_byte_array = bytearray(image_byte_stream.read())
        image_numpy_array = np.asarray(image_byte_array, dtype=np.uint8)
        image_numpy_array = cv2.imdecode(image_numpy_array, cv2.IMREAD_UNCHANGED)
        
        if channel == 'bgr':
            image_numpy_array = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2RGB)
        
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
