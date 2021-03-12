# TBD 1 : logger 추가
# TBD 2: flask github 참고, method, class, 파일의 맨 윗단 마다 pydoc 형식으로 달기
# TBD 3: 축약어를 자제할것 (특히 변수)

# -------------------------
#   To-do
# -------------------------
# 1. add logger
# 2. make image drawer overlay mask on image
# 3. make iterable
# 4. make verbose turn on and off
# 5. write pydoc

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

# tensorflow Module
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.initializers import RandomNormal

# python basic Module
import os
from datetime import datetime

import numpy as np

from gan_module.data_loader.medical_segmentation_data_loader import DataLoader

from gan_module.model.build_model import build_generator
from gan_module import custom_loss
from gan_module.custom_loss import dice_loss_for_training, f1_score
from gan_module.config import CONFIG


custom_loss.AXIS = [1, 2]
USE_GPU = True
# set GPU memory growth allocation
if USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class UnetSegmentation:
    def __init__(
        self,
        generator_power=32,
        generator_learning_rate=1e-4,
        on_memory=True,
        code_test=False
    ):

        img_shape = CONFIG["img_shape"]
        input_channels = CONFIG["input_channels"]
        output_channels = CONFIG["output_channels"]

        input_img_shape = (*img_shape, input_channels)

        # Configure data loader
        self.dataset_name = "tumor"
        self.data_loader = DataLoader(
            dataset_name=self.dataset_name,
            on_memory=on_memory, code_test=code_test,
        )

        self.loaded_data_index = {
            "train": np.arange(self.data_loader.data_length["train"]),
            "valid": np.arange(self.data_loader.data_length["valid"])
        }

        # Number of filters in the first layer of G and D
        self.generator_power = generator_power
        self.generator_learning_rate = generator_learning_rate
        generator_optimizer = Nadam(self.generator_learning_rate)

        # layer Component
        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)

        # Build the generator
        self.generator = build_generator(
            input_img_shape=input_img_shape,
            output_channels=output_channels,
            generator_power=self.generator_power,
            kernel_initializer=self.kernel_initializer,
        )
        # loss = sm.losses.bce_dice_loss
        self.generator.compile(
            loss=dice_loss_for_training,
            optimizer=generator_optimizer,
            metrics=[f1_score],
        )

    def train(self, epochs, batch_size=10, start_epoch=0):

        start_time = datetime.now()

        reduce_lr = LearningRateScheduler(self.learning_rate_scheduler)
        save_c = ModelCheckpoint(
            './U_net/U_net_{epoch:02d}.h5', monitor="loss", save_best_only=False, save_freq=1)
        csv_logger = CSVLogger('./U_net/log.csv', append=False, separator=',')

        self.generator.fit(
            x=self.data_loader.loaded_data["train"][0],
            y=self.data_loader.loaded_data["train"][1],
            validation_data=self.data_loader.loaded_data["valid"],
            batch_size=batch_size, epochs=epochs,
            callbacks=[reduce_lr, save_c, csv_logger],
            initial_epoch=start_epoch
        )

        elapsed_time = datetime.now() - start_time
        print(f"elapsed_time : {elapsed_time}")

    def learning_rate_scheduler(self, epoch,
                                schedule_list=None, exponent=0.2,
                                warm_up=True, warm_up_epoch=10):
        step = 0
        if warm_up and epoch < warm_up_epoch:
            new_learning_rate = self.generator_learning_rate * \
                ((epoch + 1) / warm_up_epoch)
        else:
            if schedule_list is None:
                schedule_list = [30, 100, 175, 250, 325]
            for step, target_epoch in enumerate(schedule_list):
                if target_epoch > epoch:
                    break
                else:
                    continue
            new_learning_rate = self.generator_learning_rate * \
                (exponent**(step))

        return new_learning_rate


if __name__ == '__main__':
    u_net = UnetSegmentation(code_test=False)
    generator_lr = 1e-3
    batch_size = 10
    g_lr = generator_lr * batch_size
    u_net.train(epochs=325, batch_size=batch_size, start_epoch=0)
