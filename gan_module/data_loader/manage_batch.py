import threading
from queue import Queue
import tensorflow as tf


class BatchQueueManager():

    def __init__(self, train_class):

        self.train_class = train_class
        self.__batch_queue = {"train": Queue(), "valid": Queue()}

        self.__batch_setter = {
            "train": threading.Thread(
                target=self.__get_batch_setter, args=("train",), daemon=True).start(),
            "valid": threading.Thread(
                target=self.__get_batch_setter, args=("valid",), daemon=True).start()
        }

    def get_batch(self, data_mode):

        original_img, masked_img = self.__batch_queue[data_mode].get()
        tensor_original_img = tf.convert_to_tensor(original_img)
        tensor_masked_img = tf.convert_to_tensor(masked_img)
        self.__batch_queue[data_mode].task_done()

        return tensor_original_img, tensor_masked_img

    def __get_batch_setter(self, data_mode):
        while True:
            self.__fill_queue(data_mode=data_mode)

    def __fill_queue(self, data_mode):
        batch_i = 0
        while (batch_i + self.train_class.batch_size
                <= self.train_class.data_loader.data_length[data_mode]):

            batch_index = self.train_class.loaded_data_index[data_mode][batch_i: batch_i +
                                                                        self.train_class.batch_size]
            batch_tuple = self.train_class.data_loader.get_data(
                data_mode=data_mode, index=batch_index)

            self.__batch_queue[data_mode].put(batch_tuple)
            self.__batch_queue[data_mode].join()
            batch_i += self.train_class.batch_size
