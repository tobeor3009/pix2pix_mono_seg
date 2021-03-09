import cv2
import numpy as np

import os
from glob import glob
from collections import deque


TEST_DATA_NUM = 20


class DataLoader:

    def __init__(self, dataset_name, img_res=(512, 512), on_memory=False, code_test=False):

        self.img_res = img_res
        self.on_memory = on_memory
        self.loaded_data = None
        # -----------------
        #  get FilePath
        # -----------------

        current_path = os.getcwd()
        dataset_path = os.path.join(current_path, "datasets", dataset_name)

        image_data_path_regexp = {
            "train": os.path.join(dataset_path, "train", "image", "*"),
            "valid": os.path.join(dataset_path, "valid", "image", "*")
        }

        mask_data_path_regexp = {
            "train": os.path.join(dataset_path, "train", "mask", "*"),
            "valid": os.path.join(dataset_path, "valid", "mask", "*")
        }

        self.image_file_paths = {
            "train": np.array(glob(image_data_path_regexp["train"])),
            "valid": np.array(glob(image_data_path_regexp["valid"]))
        }

        self.mask_file_paths = {
            "train": np.array(glob(mask_data_path_regexp["train"])),
            "valid": np.array(glob(mask_data_path_regexp["valid"]))
        }

        if code_test:
            self.image_file_paths["train"] = self.image_file_paths["train"][:TEST_DATA_NUM]
            self.mask_file_paths["train"] = self.mask_file_paths["train"][:TEST_DATA_NUM]
            self.image_file_paths["valid"] = self.image_file_paths["valid"][:TEST_DATA_NUM]
            self.mask_file_paths["valid"] = self.mask_file_paths["valid"][:TEST_DATA_NUM]

        self.data_length = {
            "train": len(self.image_file_paths["train"]),
            "valid": len(self.image_file_paths["valid"])
        }

        self.data_index = {
            "train": np.arange(self.data_length["train"]),
            "valid": np.arange(self.data_length["valid"])
        }

        # -----------------
        #  load File if on_memory
        # -----------------

        if self.on_memory:
            self.loaded_data = self.__load_whole_data()

    def get_data(self, data_mode, index=None):

        if self.on_memory:
            if index is None:
                data_tuple = self.loaded_data[data_mode]
            else:
                data_tuple = (self.loaded_data[data_mode][0][index],
                              self.loaded_data[data_mode][1][index])
        else:
            data_tuple = self.__get_processed_imgs(
                data_mode=data_mode, index=index)

        return data_tuple

    def __load_whole_data(self):
        tuple_train_data = self.__get_processed_imgs(data_mode="train")
        tuple_valid_data = self.__get_processed_imgs(data_mode="valid")

        return {"train": tuple_train_data, "valid": tuple_valid_data}

    # separate imgs and make png data range(-255,255) to (-1,1)

    def __get_processed_imgs(self, data_mode, index=None):

        image_list, mask_list = deque(), deque()

        image_paths = self.image_file_paths[data_mode]
        mask_paths = self.mask_file_paths[data_mode]

        if index is None:
            pass
        else:
            image_paths = image_paths[index]
            mask_paths = mask_paths[index]

        for image_path, mask_path in zip(image_paths, mask_paths):
            image = imread(image_path, channel='bgr')
            mask = imread(mask_path)

            image_list.append(np.array(image))
            mask_list.append(np.array(mask))

        image_list = np.array(image_list, dtype=np.float32) / 127.5 - 1.0
        mask_list = np.round(np.array(mask_list, dtype=np.float32) / 255)

        return image_list, mask_list


def imread(path, channel=None):
    image_byte_stream = open(path.encode("utf-8"), "rb")
    image_byte_array = bytearray(image_byte_stream.read())
    image_numpy_array = np.asarray(image_byte_array, dtype=np.uint8)
    image_numpy_array = cv2.imdecode(
        image_numpy_array, cv2.IMREAD_UNCHANGED)

    if channel == 'bgr':
        image_numpy_array = cv2.cvtColor(
            image_numpy_array, cv2.COLOR_BGR2RGB)

    return image_numpy_array
