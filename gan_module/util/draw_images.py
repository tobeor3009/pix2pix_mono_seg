# python basic Module
import os
import cv2

# math, plot Module
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


class ImageDrawer:
    def __init__(self, dataset_name, data_loader):
        self.dataset_name = dataset_name
        self.data_loader = data_loader

        self.plot_dir_name = "plot"
        self.plot_worst_dir_name = "plot_worst"
        self.plot_best_dir_name = "plot_best"

        # make dir
        os.makedirs(f"images/{self.dataset_name}/{self.plot_dir_name}/train", exist_ok=True)
        os.makedirs(f"images/{self.dataset_name}/{self.plot_dir_name}/valid", exist_ok=True)
        os.makedirs(f"images/{self.dataset_name}/{self.plot_worst_dir_name}", exist_ok=True)
        os.makedirs(f"images/{self.dataset_name}/{self.plot_best_dir_name}", exist_ok=True)

    def sample_images(self, generator, epoch, batch_i=None):

        train_data_index = np.random.choice(self.data_loader.data_length["train"], 3)
        valid_data_index = np.random.choice(self.data_loader.data_length["valid"], 3)

        train_original_imgs, train_masked_imgs = self.data_loader.get_data(
            data_mode="train", index=train_data_index
        )
        valid_original_imgs, valid_masked_imgs = self.data_loader.get_data(
            data_mode="valid", index=valid_data_index
        )

        train_list_of_plot_imgs = self.__get_list_of_plot_images(
            generator, train_original_imgs, train_masked_imgs
        )
        valid_list_of_plot_imgs = self.__get_list_of_plot_images(
            generator, valid_original_imgs, valid_masked_imgs
        )

        self.__draw_images(train_list_of_plot_imgs, epoch, batch_i, train=True)
        self.__draw_images(valid_list_of_plot_imgs, epoch, batch_i, train=False)

    def draw_worst_and_best(self, original_img, model_made_img, man_made_img, epoch, worst=True):

        original_img = original_img[0:1]
        model_made_img = model_made_img[0:1]
        man_made_img = man_made_img[0:1]

        model_made_img = np.array([self.__gray_to_rgb(model_made_img[0])])
        man_made_img = np.array([self.__gray_to_rgb(man_made_img[0])])
        r, c = 3, 1

        gen_imgs = np.concatenate([original_img, model_made_img, man_made_img])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ["Source", "Model", "GroundTruth"]
        fig, axs = plt.subplots(r, c)
        fig.set_size_inches(12, 12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i].imshow(gen_imgs[cnt])
                axs[i].set_title(titles[i])
                axs[i].axis("off")
                cnt += 1
        if worst:
            best_or_worst = self.plot_worst_dir_name
        else:
            best_or_worst = self.plot_best_dir_name
        fig.savefig(f"images/{self.dataset_name}/{best_or_worst}/{epoch}.png")
        plt.close()

    def __get_list_of_plot_images(self, generator, original_imgs, masked_imgs):
        print(original_imgs.shape)
        print(masked_imgs.shape)
        print(generator.predict(original_imgs).shape)
        model_masked_imgs = generator.predict(original_imgs)[:, :, :, 0]
        model_masked_rgb_imgs = []
        masked_rgb_imgs = []

        for (model_masked_img, masked_img) in zip(model_masked_imgs, masked_imgs):
            model_masked_rgb_imgs.append(self.__gray_to_rgb(model_masked_img))
            masked_rgb_imgs.append(self.__gray_to_rgb(masked_img))

        # rescale [-1,1] rgb image to [0,1] iamge
        original_imgs = (original_imgs + 1) / 2
        model_masked_rgb_imgs = np.array(model_masked_rgb_imgs)
        masked_rgb_imgs = np.array(masked_rgb_imgs)

        list_of_plot_imgs = np.stack([original_imgs, model_masked_rgb_imgs, masked_rgb_imgs])

        return list_of_plot_imgs

    def __draw_images(self, list_of_plot_images, epoch, batch_i=None, train=True, worst=None):
        image_row_num = list_of_plot_images.shape[0]
        image_col_num = list_of_plot_images.shape[1]

        if train:
            train_or_valid = "train"
        else:
            train_or_valid = "valid"

        titles = ["Source", "Model", "GroundTruth"]
        fig, axs = plt.subplots(image_row_num, image_col_num)
        # axs is 1-D array when image_col_num == 1
        if image_col_num == 1:
            axs = axs.reshape(*axs.shape, 1)

        for i in range(image_row_num):
            for j in range(image_col_num):
                axs[i, j].imshow(list_of_plot_images[i, j])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis("off")
        fig.set_size_inches(12, 12)

        if worst is None:
            if batch_i:
                fig.savefig(
                    f"images/{self.dataset_name}/{self.plot_dir_name}/{train_or_valid}/{epoch}_{batch_i}.png"
                )
            else:
                fig.savefig(
                    f"images/{self.dataset_name}/{self.plot_dir_name}/{train_or_valid}/{epoch}.png"
                )
        else:
            if worst:
                best_or_worst_dir_name = self.plot_worst_dir_name
            else:
                best_or_worst_dir_name = self.plot_best_dir_name
            fig.savefig(
                f"images/{self.dataset_name}/{best_or_worst_dir_name}/{train_or_valid}/{epoch}.png"
            )
        plt.close()

    def __gray_to_rgb(self, gray_image):
        if tf.is_tensor(gray_image):
            gray_image = gray_image.numpy()
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
