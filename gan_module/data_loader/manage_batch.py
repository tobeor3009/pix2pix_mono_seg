import threading
from queue import Queue
import tensorflow as tf
import numpy as np


class BatchQueueManager:
    def __init__(self, train_class, batch_size, on_memory):

        self.train_class = train_class
        self.batch_size = batch_size
        self.on_memory = on_memory
        self.__batch_queue = {"train": Queue(), "valid": Queue()}

        self.__batch_setter = {
            "train": threading.Thread(
                target=self.__batch_setter, args=("train",), daemon=True
            ).start(),
            "valid": threading.Thread(
                target=self.__batch_setter, args=("valid",), daemon=True
            ).start(),
        }

    def get_batch(self, data_mode):

        original_img, masked_img = self.__batch_queue[data_mode].get()
        tensor_original_img = tf.convert_to_tensor(original_img)
        tensor_masked_img = tf.convert_to_tensor(masked_img)
        self.__batch_queue[data_mode].task_done()

        return tensor_original_img, tensor_masked_img

    def __batch_setter(self, data_mode):
        while True:
            self.__fill_queue_test(data_mode=data_mode)

    def __fill_queue(self, data_mode):
        batch_i = 0
        while (
            batch_i + self.train_class.batch_size
            <= self.train_class.data_loader.data_length[data_mode]
        ):

            batch_index = self.train_class.loaded_data_index[data_mode][
                batch_i : batch_i + self.train_class.batch_size
            ]
            batch_tuple = self.train_class.data_loader.get_data(
                data_mode=data_mode, index=batch_index
            )

            self.__batch_queue[data_mode].put(batch_tuple)
            self.__batch_queue[data_mode].join()
            batch_i += self.train_class.batch_size

    def __fill_queue_test(self, data_mode):
        batch_input_img = np.empty((self.batch_size, *self.train_class.input_img_shape))
        batch_output_img = np.empty((self.batch_size, *self.train_class.output_img_shape))
        if self.on_memory:
            iter_object = enumerate(
                zip(*self.train_class.data_loader.loaded_data_object[data_mode].values())
            )
        else:
            iter_object = enumerate(self.train_class.data_loader.loaded_data_object[data_mode])
        for index, (input_img, output_img) in iter_object:
            if index > 0 and index % self.batch_size == 0:
                batch_tuple = (
                    batch_input_img.copy(),
                    batch_output_img.copy(),
                )
                self.__batch_queue[data_mode].put(batch_tuple)
                self.__batch_queue[data_mode].join()
            batch_input_img[index % self.batch_size] = input_img
            batch_output_img[index % self.batch_size] = output_img

