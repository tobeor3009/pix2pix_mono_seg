# TBD 1 : logger 추가
# TBD 2: flask github 참고, method, class, 파일의 맨 윗단 마다 pydoc 형식으로 달기
# TBD 3: 축약어를 자제할것 (특히 변수)

# -------------------------
#   done
# -------------------------

# 0. add data-setter, receiver system use python queue.Queue() class
# this will resolve i/o bottleneck
# 3. make iterable

# -------------------------
#   In Progress
# -------------------------

# 1. add logger
# 2. make image drawer overlay mask on image

# -------------------------
#   To be Done
# -------------------------

# 4. make verbose turn on and off
# 5. write pydoc

# python basic Module
import os
import sys
import types
import progressbar
from datetime import datetime
from shutil import copy
from pickle import dump, load

# math, image, plot Module
import numpy as np
import cv2
import matplotlib.pyplot as plt  # TBD

# tensorflow Module
import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import losses

# keras segmentaion third-party Moudle
import segmentation_models as sm
import tensorflow_addons as tfa

# custom Module
from gan_module.data_loader.medical_segmentation_data_loader import DataLoader
from gan_module.data_loader.manage_batch import BatchQueueManager

from gan_module.model.build_model import build_generator_non_unet as build_generator
from gan_module.model.build_model import build_discriminator as build_discriminator
from gan_module.util.custom_loss import dice_loss_for_training, dice_score
# from gan_module.util.custom_gradient import SGD_AGC
from gan_module.util.manage_learning_rate import learning_rate_scheduler
from gan_module.util.draw_images import ImageDrawer
from gan_module.util.logger import TrainLogger
from gan_module.config import CONFIG

USE_GPU = True

if USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Pix2PixSegmentation:
    def __init__(
        self,
        generator_power=32,
        discriminator_power=32,
        generator_depth=None,
        discriminator_depth=None,
        generator_learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
        temp_weights_path=".",
        on_memory=True,
        code_test=False
    ):
        # Input shape
        img_shape = CONFIG["img_shape"]
        input_channels = CONFIG["input_channels"]
        output_channels = CONFIG["output_channels"]

        self.input_img_shape = (*img_shape, input_channels)
        self.output_img_shape = (*img_shape, output_channels)
        # set parameter
        self.start_epoch = None
        self.on_memory = on_memory
        self.history = {"generator_loss": [],
                        "f1_loss_train": [], "f1_score_train": [],
                        "f1_loss_valid": [], "f1_score_valid": []}
        self.temp_weights_path = temp_weights_path

        # Configure data loader
        self.dataset_name = "glomerulus_0.65_512_not_filped"
        self.data_loader = DataLoader(
            dataset_name=self.dataset_name,
            config_dict=CONFIG,
            on_memory=self.on_memory,
            code_test=code_test
        )

        self.train_logger = TrainLogger()

        self.loaded_data_index = {
            "train": np.arange(self.data_loader.data_length["train"]),
            "valid": np.arange(self.data_loader.data_length["valid"])
        }

        # Configure Image Drawer
        self.image_drawer = ImageDrawer(
            dataset_name=self.dataset_name, data_loader=self.data_loader
        )
        self.discriminator_loss_ratio = keras_backend.variable(0.1)
        self.f1_loss_ratio = keras_backend.variable(100)
        self.discriminator_losses = np.array(
            [1 for _ in range(self.data_loader.data_length["train"])], dtype=np.float32)
        self.discriminator_acc_previous = 0.5
        self.discriminator_acces = np.array(
            [0.5 for _ in range(self.data_loader.data_length["train"])])
        self.discriminator_acces_previous = self.discriminator_acces.copy()
        self.generator_losses = np.array(
            [1 for _ in range(self.data_loader.data_length["train"])], dtype=np.float32)
        self.generator_losses_previous = self.generator_losses.copy()
        self.generator_f1_losses = np.array(
            [1 for _ in range(self.data_loader.data_length["train"])], dtype=np.float32)
        self.generator_loss_min = 1000
        self.generator_loss_previous = 1000
        self.generator_loss_max_previous = 1000
        self.total_f1_loss_min = 2
        self.weight_save_stack = False
        self.training_end_stack = 0
        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (img_shape[0] // (2 ** discriminator_depth),
                           img_shape[1] // (2 ** discriminator_depth), 1)
        # Number of filters in the first layer of G and D
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.patience_count = 0

        generator_optimizer = Nadam(self.generator_learning_rate)
        discriminator_optimizer = Nadam(self.discriminator_learning_rate)
#         generator_optimizer = SGD_AGC(lr=self.generator_learning_rate, momentum=0.9)
#         discriminator_optimizer = SGD_AGC(lr=self.discriminator_learning_rate, momentum=0.9)
        # Build the generator
        self.generator = build_generator(
            input_img_shape=self.input_img_shape,
            output_channels=output_channels,
            generator_power=generator_power,
            depth=generator_depth,
        )
        self.generator.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=generator_optimizer,
            metrics=[dice_score],
        )
        # Build and compile the discriminator
        self.discriminator = build_discriminator(
            input_img_shape=self.input_img_shape,
            output_img_shape=self.output_img_shape,
            discriminator_power=discriminator_power,
            depth=discriminator_depth,
        )
        # 'mse' or tf.keras.losses.Huber() tf.keras.losses.LogCosh()
        self.discriminator.compile(
            loss=sm.losses.BinaryFocalLoss(alpha=0.25, gamma=4),
            optimizer=discriminator_optimizer,
            metrics=["accuracy"],
        )
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Input images and their conditioning images
        original_img = Input(shape=self.input_img_shape)
        # generate image from original_img for target masked_img
        model_masked_img = self.generator(original_img)

        # Discriminators determines validity of translated images / condition pairs
        model_validity = self.discriminator([original_img, model_masked_img])
        # give score by
        # 1. how generator trick discriminator
        # 2. how generator's image same as real photo in pixel
        # 3. if you want change loss, see doc https://keras.io/api/losses/
        # 4. 'mse', 'mae', tf.keras.losses.LogCosh(),  tf.keras.losses.Huber()
        self.combined = Model(
            inputs=original_img,
            outputs=[model_validity, model_masked_img],
        )

        self.combined.compile(
            loss=[
#                 tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                sm.losses.BinaryFocalLoss(alpha=0.25, gamma=4),
                dice_loss_for_training
            ],
            loss_weights=[0.1, 100],
            optimizer=generator_optimizer,
        )

    def train(self, epochs, batch_size=1, epoch_shuffle_term=10):

        start_time = datetime.now()

        # Adversarial loss ground truths
        self.training_end_stack = 0
        self.batch_size = batch_size
        valid_patch = np.ones(
            (self.batch_size, *self.disc_patch), dtype=np.float32)
        fake_patch = np.zeros(
            (self.batch_size, *self.disc_patch), dtype=np.float32)
        # TBD : move batch_queue_manager to __init__
        self.batch_queue_manager = BatchQueueManager(
            self, batch_size, self.on_memory)

        if self.start_epoch is None:
            self.start_epoch = 0
        for epoch in range(self.start_epoch, epochs):
            batch_i = 0

            generator_loss_max_in_epoch = 0
            generator_loss_min_in_epoch = 1000
            generator_discriminator_losses = np.array(
            [1 for _ in range(self.data_loader.data_length["train"])], dtype=np.float32)
            # shffle data maybe
            if epoch % epoch_shuffle_term == 0:
                self.data_loader.shuffle_train_imgs()

            if self.discriminator_acc_previous < 0.9:
                discriminator_learning = True
            else:
                discriminator_learning = False
            generator_1_10_quantile = np.quantile(self.generator_losses, 0.1)

            generator_current_learning_rate = learning_rate_scheduler(
                self.generator_learning_rate,
                epoch + self.patience_count,
                warm_up=True
            )
            discriminator_current_learning_rate = learning_rate_scheduler(
                self.discriminator_learning_rate,
                epoch + self.patience_count,
                warm_up=True
            ) * (1 - self.discriminator_acc_previous)
            keras_backend.set_value(
                self.discriminator.optimizer.learning_rate,
                discriminator_current_learning_rate,
            )
            keras_backend.set_value(
                self.discriminator.optimizer.learning_rate,
                discriminator_current_learning_rate,
            )
            keras_backend.set_value(
                self.discriminator_loss_ratio,
                keras_backend.variable(0.01) + 1 *
                                       self.discriminator_acc_previous,
            )
            keras_backend.set_value(
                self.f1_loss_ratio,
                keras_backend.variable(100) - 1 *
                                       self.discriminator_acc_previous,
            )

            bar = progressbar.ProgressBar(
                maxval=self.data_loader.data_length["train"]).start()

            while batch_i + self.batch_size <= self.data_loader.data_length["train"]:

                batch_index = self.loaded_data_index["train"][batch_i: batch_i +
                                                              self.batch_size]

                original_img, masked_img = self.batch_queue_manager.get_batch(
                    data_mode="train")
                model_masked_img = self.generator.predict_on_batch(
                    original_img)

                valid_patch = np.ones(
                    (len(model_masked_img), *self.disc_patch), dtype=np.float32)
                fake_patch = np.zeros(
                    (len(model_masked_img), *self.disc_patch), dtype=np.float32)

                self.original_img = original_img
                self.masked_img = masked_img
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Train Discriminator for valid image if it failed to detect fake image
                if discriminator_learning and self.discriminator_acc_previous < np.random.rand():
                    discriminator_loss = self.discriminator.train_on_batch(
                        [original_img, masked_img], valid_patch)
                else:
                    discriminator_loss = self.discriminator.test_on_batch(
                        [original_img, masked_img], valid_patch)

                batch_discriminator_acc_previous = np.mean(
                    self.discriminator_acces_previous[batch_index])
                self.discriminator.trainable = False
                # -----------------
                #  Train Generator
                # -----------------

                if self.generator_losses[batch_index] >= generator_1_10_quantile:
                    generator_loss = self.combined.train_on_batch(
                        original_img,
                        [valid_patch, masked_img]
                    )
                else:
                    generator_loss = self.combined.test_on_batch
                        original_img,
                        [valid_patch, masked_img]
                    )
                # train discriminator for fake image if it failed to detect fake image
                self.discriminator.trainable=True
                if (batch_discriminator_acc_previous <= 0.5 or epoch == 0) and discriminator_learning:
                    discriminator_loss += self.discriminator.train_on_batch(
                        [original_img, model_masked_img], fake_patch)
                else:
                    discriminator_loss += self.discriminator.test_on_batch(
                        [original_img, model_masked_img], fake_patch)

                self.discriminator_losses[batch_index]=discriminator_loss[0]
                self.discriminator_acces[batch_index]=discriminator_loss[1]
                self.generator_losses[batch_index]=generator_loss[0]
                self.generator_f1_losses[batch_index]=generator_loss[2]
                generator_discriminator_losses=generator_loss[1]
                # plot progress
                bar.update(batch_i)

                # 한 배치 끝
                batch_i += self.batch_size


            # training batch 사이클 끝

            self.train_logger.write_log(
                f"{epoch}/{epochs} ({epoch+self.patience_count})",
                np.mean(self.discriminator_acces),
                np.mean(self.generator_losses),
                np.max(self.generator_losses),
                np.min(self.generator_losses),
                f"{self.generator_loss_min - np.mean(self.generator_losses)}({np.mean(self.generator_losses) / self.generator_loss_min})",
                self.generator_loss_min,
                generator_current_learning_rate,
                datetime.now() - start_time
            )
            self.image_drawer.sample_images(
                self.generator, epoch)

            if np.mean(self.generator_losses) / self.generator_loss_min < 1.05:
                # if self.generator_loss_min > np.mean(self.generator_losses):
                valid_f1_loss_list=[]
                for index in range(0, self.data_loader.data_length["valid"], self.batch_size):

                    valid_source_img, valid_masked_img=self.batch_queue_manager.get_batch(
                        data_mode = "valid")

                    valid_model_masked_img=self.generator.predict_on_batch(
                        valid_source_img)
                    valid_f1_loss=dice_score(
                        valid_masked_img, valid_model_masked_img)
                    valid_f1_loss_list.append(valid_f1_loss)
                # compute valid_f1_loss end
                total_f1_loss=np.mean(
                    self.generator_f1_losses) + np.mean(valid_f1_loss_list)

                print(
                    f"discriminator_loss : {np.mean(self.discriminator_losses)}")
                print(
                    f"generator_discriminator_loss : {np.mean(generator_discriminator_losses)}")
                print(f"train_f1_loss : {np.mean(self.generator_f1_losses)}")
                print(f"valid_f1_loss : {np.mean(valid_f1_loss_list)}")
                print(
                    f"current/min total_f1_loss = {total_f1_loss} / {self.total_f1_loss_min}")
                if self.generator_loss_min > np.mean(self.generator_losses):
                    self.generator_loss_min= np.mean(self.generator_losses)
                    self.generator_loss_max_min= generator_loss_max_in_epoch
                    self.generator_loss_min_min= generator_loss_min_in_epoch
                    self.save_study_info()
                    self.weight_save_stack= True
                    print("save weights")
                if self.total_f1_loss_min > total_f1_loss:
                    self.total_f1_loss_min= total_f1_loss


            else:
                print("loss decrease.")
                self.patience_count += 5
                self.load_best_weights()

            # previous generator_loss 갱신
            self.generator_loss_previous= np.mean(self.generator_losses)
            self.generator_loss_max_previous= np.max(self.generator_losses)

            if epoch >= 10 and self.weight_save_stack:
                copy(
                    "generator.h5",
                    "./generator_weights/generator_"
                    + str(round(self.generator_loss_min, 5))
                    + "_"
                    + str(round(self.generator_loss_max_min, 5))
                    + ".h5",
                )
                self.weight_save_stack = False

            self.discriminator_acc_previous = np.mean(self.discriminator_acces)
            self.discriminator_acces_previous = self.discriminator_acces.copy()
            self.generator_losses_previous = self.generator_losses.copy()
            # TBD: add epoch bigger than history length
            self.history["generator_loss"].append(
                np.mean(self.generator_losses))
            self.history["f1_loss_train"].append(
                np.mean(self.generator_f1_losses))
            self.history["f1_loss_valid"].append(
                np.mean(valid_f1_loss_list))

    def get_info_folderPath(self):
        return (
            str(round(self.generator_loss_min, 5))
            + "_"
            + str(round(self.generator_loss_max_min, 5))
        )

    def save_study_info(self, path=None):

        if path is None:
            path = self.temp_weights_path

        generator_weigth_path = os.path.join(path, "generator.h5")
        discriminator_weigth_path = os.path.join(path, "discriminator.h5")
        combined_weigth_path = os.path.join(path, "combined.h5")

        self.generator.save_weights(generator_weigth_path)
        self.discriminator.save_weights(discriminator_weigth_path)
        self.combined.save_weights(combined_weigth_path)

        study_info = {}
        study_info["start_epoch"] = self.start_epoch
        study_info["train_loaded_data_index"] = self.loaded_data_index["train"]
        study_info["generator_loss_min"] = self.generator_loss_min
        study_info["generator_loss_max_min"] = self.generator_loss_max_min
        study_info["generator_loss_min_min"] = self.generator_loss_min_min
        study_info["generator_losses_previous"] = self.generator_losses_previous
        study_info["discriminator_acces"] = self.discriminator_acces
        study_info["history"] = self.history
        file = open(path + "/study_info.pkl", "wb")
        dump(study_info, file)
        file.close()

    def load_study_info(self):

        self.generator.load_weights("generator.h5")
        self.discriminator.load_weights("discriminator.h5")
#         self.combined.load_weights("combined.h5")

        if os.path.isfile("study_info.pkl"):
            file = open("study_info.pkl", "rb")
            study_info = load(file)
            file.close()
            self.start_epoch = study_info["start_epoch"]
            self.loaded_data_index["train"] = study_info["train_loaded_data_index"]
            self.generator_loss_min = study_info["generator_loss_min"]
            self.generator_loss_max_min = study_info["generator_loss_max_min"]
            self.generator_loss_min_min = study_info["generator_loss_min_min"]
            self.generator_losses_previous = study_info["generator_losses_previous"]
            self.discriminator_acces = study_info["discriminator_acces"]
            self.history = study_info["history"]
        else:
            print("No info pkl file!")

    def load_best_weights(self):
        self.generator.load_weights(self.temp_weights_path + "/generator.h5")
        self.discriminator.load_weights(
            self.temp_weights_path + "/discriminator.h5")
        self.combined.load_weights(self.temp_weights_path + "/combined.h5")

    def run_pretraining(self, epochs):
        if self.on_memory:
            self.generator.fit(
                x=self.data_loader.loaded_data_object["train"]["input"],
                y=self.data_loader.loaded_data_object["train"]["output"],
                validation_data=list(self.data_loader.loaded_data_object["valid"].values()),
                batch_size=self.batch_size, epochs=epochs
            )
        else:
            self.generator.fit_generator(
                x=self.data_loader.loaded_data_object["train"]["input"],
                y=self.data_loader.loaded_data_object["train"]["output"],
                validation_data=list(self.data_loader.loaded_data_object["valid"].values()),
                batch_size=self.batch_size, epochs=epochs
            )
        self.generator.save_weights("pretrained.h5")
