# TBD 1 : logger 추가
# TBD 2: flask github 참고, method, class, 파일의 맨 윗단 마다 pydoc 형식으로 달기
# TBD 3: 축약어를 자제할것 (특히 변수)

# -------------------------
#   To-do
# -------------------------

# 0. add data-setter, receiver system use python queue.Queue() class
# this will resolve i/o bottleneck
# 1. add logger
# 2. make image drawer overlay mask on image
# 3. make iterable
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

# custom Module
from gan_module.data_loader.medical_segmentation_data_loader_v3 import DataLoader
from gan_module.data_loader.manage_batch import BatchQueueManager

from gan_module.model.build_model import build_generator, build_discriminator
from gan_module.util.draw_images import ImageDrawer
from gan_module.custom_loss import f1_loss_for_training, f1_score, dice_loss_for_training, jaccard_coef_loss_for_training
from gan_module.util.manage_learning_rate import learning_rate_scheduler

gpu_on = True

if gpu_on:
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
        generator_learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
        find_error=False,
        temp_weights_path=".",
        draw_images=True,
        on_memory=True,
        test=False
    ):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.input_channels = 3
        self.output_channels = 1
        self.input_img_shape = (
            self.img_rows, self.img_cols, self.input_channels)
        self.output_img_shape = (
            self.img_rows, self.img_cols, self.output_channels)
        # set parameter
        self.start_epoch = None
        self.history = {"generator_loss": [],
                        "f1_score_train": [], "f1_score_valid": []}
        self.f1_loss_ratio = 100
        self.find_error = find_error
        self.find_error_epoch = 30
        self.error_list = []
        self.temp_weights_path = temp_weights_path

        # Configure data loader
        self.dataset_name = "tumor"
        self.data_loader = DataLoader(
            dataset_name=self.dataset_name,
            img_res=(self.img_rows, self.img_cols),
            on_memory=on_memory, test=test
        )

        self.loaded_data_index = {
            "train": np.arange(self.data_loader.data_length["train"]),
            "valid": np.arange(self.data_loader.data_length["valid"])
        }

        # Configure Image Drawer
        self.draw_images = draw_images
        self.image_drawer = ImageDrawer(
            dataset_name=self.dataset_name, data_loader=self.data_loader
        )
        self.discriminator_acc_previous = 0.5
        self.discriminator_acces = np.array(
            [0.5 for _ in range(self.data_loader.data_length["train"])])
        self.discriminator_acces_previous = self.discriminator_acces.copy()
        self.generator_losses = np.array(
            [1 for _ in range(self.data_loader.data_length["train"])], dtype=np.float32)
        self.generator_losses_previous = self.generator_losses.copy()
        self.generator_loss_min = 500
        self.generator_loss_previous = 100
        self.generator_loss_max_previous = 1000
        self.generator_loss_max_min = 1000
        self.generator_loss_min_min = 1000
        self.weight_save_stack = False
        self.training_end_stack = 0
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 2)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.generator_power = generator_power
        self.discriminator_power = discriminator_power
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        generator_optimizer = Nadam(self.generator_learning_rate)
        discriminator_optimizer = Nadam(self.discriminator_learning_rate)

        # layer Component
        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)

        # Build the generator
        self.generator = build_generator(
            input_img_shape=self.input_img_shape,
            output_channels=self.output_channels,
            generator_power=self.generator_power,
            kernel_initializer=self.kernel_initializer,
        )
        # Build and compile the discriminator
        self.discriminator = build_discriminator(
            input_img_shape=self.input_img_shape,
            output_img_shape=self.output_img_shape,
            discriminator_power=self.discriminator_power,
            kernel_initializer=self.kernel_initializer,
        )
        # self.discriminator = self.build_discriminator()
        # 'mse' or tf.keras.losses.Huber() tf.keras.losses.LogCosh()
        self.discriminator.compile(
            loss=tf.keras.losses.LogCosh(),
            optimizer=discriminator_optimizer,
            metrics=["accuracy"],
        )

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Input images and their conditioning images
        original_img = Input(shape=self.input_img_shape)
        masked_img = Input(shape=self.output_img_shape)
        # generate image from original_img for target masked_img
        model_masked_img = self.generator(original_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminators determines validity of translated images / condition pairs
        model_validity = self.discriminator([original_img, model_masked_img])
        # give score by
        # 1. how generator trick discriminator
        # 2. how generator's image same as real photo in pixel
        # 3. if you want change loss, see doc https://keras.io/api/losses/
        # 4. 'mse', 'mae', tf.keras.losses.LogCosh(),  tf.keras.losses.Huber()
        self.combined = Model(
            inputs=[original_img, masked_img],
            outputs=[model_validity, model_masked_img],
        )
        self.combined.compile(
            loss=[
                tf.keras.losses.LogCosh(),
                dice_loss_for_training
            ],
            loss_weights=[1, self.f1_loss_ratio],
            optimizer=generator_optimizer
        )

    def train(self, epochs, batch_size=1, sample_interval=50, epoch_shuffle_term=10):

        start_time = datetime.now()

        # Adversarial loss ground truths
        self.training_end_stack = 0
        self.batch_size = batch_size
        valid_patch = np.ones((self.batch_size,) +
                              self.disc_patch, dtype=np.float32)
        fake_patch = np.zeros((self.batch_size,) +
                              self.disc_patch, dtype=np.float32)
        # TBD : move batch_queue_manager to __init__
        self.batch_queue_manager = BatchQueueManager(self, include_valid=True)

        if self.start_epoch is None:
            self.start_epoch = 0
        for epoch in range(self.start_epoch, epochs):
            bar = progressbar.ProgressBar(
                maxval=self.data_loader.data_length["train"]).start()
            batch_i = 0

            discriminator_losses = []
            generator_loss_max_in_epoch = 0
            generator_loss_min_in_epoch = 1000

            # shffle data maybe
            if epoch % epoch_shuffle_term == 0:
                np.random.shuffle(self.loaded_data_index["train"])

            if self.discriminator_acc_previous < 0.75:
                discriminator_learning = True
                print("discriminator_learning is True")
            else:
                discriminator_learning = False
                print("discriminator_learning is False")

            while batch_i + self.batch_size <= self.data_loader.data_length["train"]:
                bar.update(batch_i)

                batch_index = self.loaded_data_index["train"][batch_i: batch_i +
                                                              self.batch_size]

                original_img, masked_img, test_batch = self.batch_queue_manager.get_batch(
                    data_mode="train")
                print(test_batch)
                model_masked_img = self.generator.predict_on_batch(
                    original_img)

                # forTest
                self.masked_img = masked_img
                self.original_img = original_img
                self.model_masked_img = model_masked_img
                self.valid_path = valid_patch
                self.fake_patch = fake_patch

                generator_current_learning_rate = learning_rate_scheduler(
                    self.generator_learning_rate,
                    epoch + 1,
                )
                discriminator_current_learning_rate = learning_rate_scheduler(
                    self.discriminator_learning_rate,
                    epoch + 1,
                )
                keras_backend.set_value(
                    self.discriminator.optimizer.learning_rate,
                    discriminator_current_learning_rate,
                )
                keras_backend.set_value(
                    self.combined.optimizer.learning_rate, generator_current_learning_rate
                )
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Train Discriminator for valid image if it failed to detect fake image

                if discriminator_learning:
                    self.discriminator.train_on_batch(
                        [original_img, masked_img], valid_patch)

                batch_discriminator_acc_previous = np.mean(
                    self.discriminator_acces_previous[batch_index])

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators

                generator_loss = self.combined.train_on_batch(
                    [original_img, masked_img],
                    [valid_patch, masked_img],
                )
                # train discriminator for fake image if it failed to detect fake image
                if (batch_discriminator_acc_previous <= 0.5 or epoch == 0) and discriminator_learning:
                    discriminator_loss = self.discriminator.train_on_batch(
                        [original_img, model_masked_img], fake_patch)
                else:
                    discriminator_loss = self.discriminator.test_on_batch(
                        [original_img, model_masked_img], fake_patch)

                self.discriminator_acces[batch_index] = discriminator_loss[1]
                self.generator_losses[batch_index] = generator_loss[0]
                elapsed_time = datetime.now() - start_time
                if batch_i % sample_interval == 0:
                    # Plot the progress
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s"
                        % (
                            epoch,
                            epochs,
                            batch_i,
                            self.data_loader.data_length["train"],
                            discriminator_loss[0],
                            100 * discriminator_loss[1],
                            generator_loss[0],
                            elapsed_time,
                        )
                    )

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.image_drawer.sample_images(
                        self.generator, epoch, batch_i)

                discriminator_losses.append(discriminator_loss[0])
                # loss 가 가장 높은 이미지를 저장 및 max_in_epoch 갱신
                if generator_loss[0] > generator_loss_max_in_epoch:
                    generator_loss_max_in_epoch = generator_loss[0]
                    if self.draw_images:
                        model_masked_img = self.generator.predict_on_batch(
                            original_img)
                        self.image_drawer.draw_worst_and_best(
                            original_img,
                            model_masked_img,
                            masked_img,
                            epoch,
                            worst=True,
                        )

                # loss 가 가장 낮은 이미지를 저장 및 max_in_epoch 갱신
                if generator_loss_min_in_epoch > generator_loss[0]:
                    generator_loss_min_in_epoch = generator_loss[0]
                    if self.draw_images:
                        model_masked_img = self.generator.predict_on_batch(
                            original_img)
                        self.image_drawer.draw_worst_and_best(
                            original_img,
                            model_masked_img,
                            masked_img,
                            epoch,
                            worst=False,
                        )

                # 한 배치 끝
                batch_i += self.batch_size
            # training batch 사이클 끝
            print(f"discriminator_acces : {np.mean(self.discriminator_acces)}")
            print(
                f"Mean generator_loss : {np.mean(self.generator_losses)}")
            print(f"Max generator_loss : {np.max(self.generator_losses)}")
            print(f"Min generator_loss : {np.min(self.generator_losses)}")
            print(
                f"generator loss decrease : {self.generator_loss_min - np.mean(self.generator_losses)}"
            )
            print(
                f"generator loss decrease ratio : ({np.mean(self.generator_losses) / self.generator_loss_min})"
            )
            print(
                f"Max generator loss decrease : {self.generator_loss_max_previous - np.max(self.generator_losses)}"
            )
            print(
                f"current lowest generator loss : {self.generator_loss_min}")
            print(
                f"current Learning_rate = {generator_current_learning_rate}")
            # rollback if loss not converge
            if np.mean(self.generator_losses) / self.generator_loss_min < 1.1:
                if self.generator_loss_min > np.mean(self.generator_losses):
                    self.generator_loss_min = np.mean(self.generator_losses)
                    self.generator_loss_max_min = generator_loss_max_in_epoch
                    self.generator_loss_min_min = generator_loss_min_in_epoch
                    self.weight_save_stack = True
                    self.save_study_info()
                    print("save weights")

                valid_f1_loss_list = []
                valid_f1_score_list = []
                valid_predict_mini_batch_size = 1
                for index in range(0, self.data_loader.data_length["valid"], valid_predict_mini_batch_size):

                    valid_source_img, valid_masked_img = self.batch_queue_manager.get_batch(
                        data_mode="valid")

                    valid_model_masked_img = self.generator.predict_on_batch(
                        valid_source_img)

                    valid_f1_loss = f1_loss_for_training(
                        valid_masked_img, np.squeeze(valid_model_masked_img))
                    valid_f1_score = f1_score(
                        valid_masked_img, np.squeeze(valid_model_masked_img))
                    valid_f1_loss_list.append(valid_f1_loss)
                    valid_f1_score_list.append(valid_f1_score)

                print(
                    f"valid_f1_loss : {np.mean(valid_f1_loss_list) * self.f1_loss_ratio}")
                print(f"valid_f1_score : {1 - np.mean(valid_f1_loss_list)}")
                print(
                    f"valid_f1_rounded_score : {np.mean(valid_f1_score_list)}")
            else:
                print("loss decrease.")
                self.load_best_weights()

            # previous generator_loss 갱신
            self.generator_loss_previous = np.mean(self.generator_losses)
            self.generator_loss_max_previous = generator_loss_max_in_epoch

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
#             if len(self.history["generator_loss"]) == epoch:
#                 self.history["generator_loss"].append(
#                     np.mean(self.generator_losses))
#                 self.history["f1_score_train"].append(
#                     np.mean(self.train_f1_score_list))
#                 self.history["f1_score_valid"].append(
#                     np.mean(self.valid_f1_score_list))
#             elif len(self.history["generator_loss"]) < epoch:
#                 self.history["generator_loss"][epoch] = np.mean(
#                     self.generator_losses)
#                 self.history["f1_score_train"][epoch] = np.mean(
#                     train_f1_score_list)
#                 self.history["f1_score_valid"][epoch] = np.mean(
#                     valid_f1_score_list)

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
        study_info["generator_loss_min"] = self.generator_loss_min
        study_info["generator_loss_max_min"] = self.generator_loss_max_min
        study_info["generator_loss_min_min"] = self.generator_loss_min_min
        study_info["generator_losses_previous"] = self.generator_losses_previous
        study_info["discriminator_acces"] = self.discriminator_acces
        study_info["history"] = self.history
        study_info["train_loaded_data_index"] = self.loaded_data_index["train"]
        file = open(path + "/study_info.pkl", "wb")
        dump(study_info, file)
        file.close()

    def load_study_info(self):

        self.generator.load_weights("generator.h5")
        self.discriminator.load_weights("discriminator.h5")
        self.combined.load_weights("combined.h5")

        if os.path.isfile("study_info.pkl"):
            file = open("study_info.pkl", "rb")
            study_info = load(file)
            file.close()
            self.start_epoch = study_info["start_epoch"]
            self.generator_loss_min = study_info["generator_loss_min"]
            self.generator_loss_max_min = study_info["generator_loss_max_min"]
            self.generator_loss_min_min = study_info["generator_loss_min_min"]
            self.generator_losses_previous = study_info["generator_losses_previous"]
            self.discriminator_acces = study_info["discriminator_acces"]
            self.history = study_info["history"]
            self.loaded_data_index["train"] = study_info["train_loaded_data_index"]
        else:
            print("No info pkl file!")

    def load_best_weights(self):
        self.generator.load_weights(self.temp_weights_path + "/generator.h5")
        self.discriminator.load_weights(
            self.temp_weights_path + "/discriminator.h5")
        self.combined.load_weights(self.temp_weights_path + "/combined.h5")
