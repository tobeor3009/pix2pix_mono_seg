import os
from glob import glob
import cv2
import numpy as np

TEST_DATA_NUM = 20


class DataLoader:
    def __init__(self, dataset_name, config_dict, on_memory=False, code_test=False):
        self.on_memory = on_memory
        self.config_dict = config_dict
        self.loaded_data = None
        # -----------------
        #  get FilePath
        # -----------------
        current_path = os.getcwd()
        dataset_path = os.path.join(current_path, "datasets", dataset_name)
        image_data_path_regexp = {
            "train": os.path.join(dataset_path, "train", "image", "*"),
            "valid": os.path.join(dataset_path, "valid", "image", "*"),
        }
        mask_data_path_regexp = {
            "train": os.path.join(dataset_path, "train", "mask", "*"),
            "valid": os.path.join(dataset_path, "valid", "mask", "*"),
        }
        self.image_file_paths = {
            "train": np.array(glob(image_data_path_regexp["train"])),
            "valid": np.array(glob(image_data_path_regexp["valid"])),
        }
        self.mask_file_paths = {
            "train": np.array(glob(mask_data_path_regexp["train"])),
            "valid": np.array(glob(mask_data_path_regexp["valid"])),
        }
        if code_test:
            self.image_file_paths["train"] = self.image_file_paths["train"][:TEST_DATA_NUM]
            self.mask_file_paths["train"] = self.mask_file_paths["train"][:TEST_DATA_NUM]
            self.image_file_paths["valid"] = self.image_file_paths["valid"][:TEST_DATA_NUM]
            self.mask_file_paths["valid"] = self.mask_file_paths["valid"][:TEST_DATA_NUM]
        self.data_length = {
            "train": len(self.image_file_paths["train"]),
            "valid": len(self.image_file_paths["valid"]),
        }
        self.data_index = {
            "train": np.arange(self.data_length["train"]),
            "valid": np.arange(self.data_length["valid"]),
        }
        # -----------------
        #  load File Object (array or generator)
        # -----------------
        self.loaded_data_object = self.__get_data_object()

    def get_data(self, data_mode, index=None):
        data_tuple = self.__get_processed_imgs(
            data_mode=data_mode, index=index).values()
        return data_tuple

    def shuffle_train_imgs(self):
        np.random.shuffle(self.data_index["train"])
        if self.on_memory:
            self.loaded_data_object["train"]["input"] = self.loaded_data_object["train"]["input"][
                self.data_index["train"]
            ]
            self.loaded_data_object["train"]["output"] = self.loaded_data_object["train"]["output"][
                self.data_index["train"]
            ]
        else:
            self.image_file_paths["train"] = self.image_file_paths["train"][
                self.data_index["train"]
            ]
            self.mask_file_paths["train"] = self.mask_file_paths["train"][
                self.data_index["train"]
            ]

    def __get_data_object(self):
        if self.on_memory:
            dict_train_data = self.__get_processed_imgs(data_mode="train")
            dict_valid_data = self.__get_processed_imgs(data_mode="valid")
            return {"train": dict_train_data, "valid": dict_valid_data}
        else:
            generator_train_data = self.__get_generator_of_processed_imgs(
                data_mode="train")
            generator_valid_data = self.__get_generator_of_processed_imgs(
                data_mode="valid")
            return {"train": generator_train_data, "valid": generator_valid_data}

    # separate imgs and make png data range(-255,255) to (-1,1)

    def __get_processed_imgs(self, data_mode, index=None):
        image_paths = self.image_file_paths[data_mode]
        mask_paths = self.mask_file_paths[data_mode]
        if index is None:
            pass
        else:
            image_paths = image_paths[index]
            mask_paths = mask_paths[index]
        input_array_shape = (
            len(image_paths),
            *self.config_dict["img_shape"],
            self.config_dict["input_channels"],
        )
        output_array_shape = (
            len(mask_paths),
            *self.config_dict["img_shape"],
            self.config_dict["output_channels"],
        )

        # this data type will redefined in batch_queue_manager
        input_image_list = np.empty(input_array_shape, dtype=np.float32)
        output_image_list = np.empty(output_array_shape, dtype=np.float32)
        for img_index, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            image = imread(image_path, channel="bgr")
            mask = imread(mask_path)
            input_image_list[img_index] = image
            output_image_list[img_index] = mask
        input_image_list = input_image_list / 127.5 - 1.0
        output_image_list = np.round(output_image_list / 255)
        return {"input": input_image_list, "output": output_image_list}

    def __get_generator_of_processed_imgs(self, data_mode):
        image_paths = self.image_file_paths[data_mode]
        mask_paths = self.mask_file_paths[data_mode]
        while True:
            path_iter = zip(image_paths, mask_paths)
            for image_path, mask_path in path_iter:
                input_image = imread(image_path, channel="bgr")
                output_image = imread(mask_path)
                # this data type will redefined in batch_queue_manager
                input_image = (input_image / 127.5 - 1.0).astype(np.float32)
                output_image = np.round(output_image / 255).astype(np.float32)
                yield input_image, output_image


def imread(path, channel=None):
    image_byte_stream = open(path.encode("utf-8"), "rb")
    image_byte_array = bytearray(image_byte_stream.read())
    image_numpy_array = np.asarray(image_byte_array, dtype=np.uint8)
    image_numpy_array = cv2.imdecode(image_numpy_array, cv2.IMREAD_UNCHANGED)
    if channel == "bgr":
        image_numpy_array = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2RGB)
    if len(image_numpy_array.shape) == 2:
        image_numpy_array = np.expand_dims(image_numpy_array, axis=-1)
    return image_numpy_array


class DataLoaderInfo:
    def __init__(self, dataset_name, data_mode):
        current_path = os.getcwd()
        dataset_path = os.path.join(current_path, "datasets", dataset_name)
        data_path_regexp = {
            "image": os.path.join(dataset_path, data_mode, "image", "*"),
            "mask": os.path.join(dataset_path, data_mode, "mask", "*"),
        }
        self.image_file_paths = {
            "image": np.array(glob(data_path_regexp["image"])),
            "mask": np.array(glob(data_path_regexp["mask"])),
        }
        self.data_length = len(self.image_file_paths["image"]["train"])
        self.data_index = np.arange(self.data_length)
