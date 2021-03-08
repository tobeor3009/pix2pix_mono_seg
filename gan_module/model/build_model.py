# tensorflow Module
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GaussianNoise
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal

# user Module
from .layers import conv2d_bn, residual_block, wide_residual_block, residual_block_last, deconv2d

DEFAULT_INITIALIZER = RandomNormal(mean=0.0, stddev=0.02)
KERNEL_SIZE = (3, 3)


def build_generator(
    input_img_shape, output_channels=4, generator_power=32, kernel_initializer=DEFAULT_INITIALIZER,
):
    """U-Net Generator"""
    # this model output range [-1, 1]. control by ResidualLastBlock's sigmiod activation

    # Image input
    input_img = Input(shape=input_img_shape)
    input_img_noised = GaussianNoise(0.1)(input_img)

    d0_1 = conv2d_bn(
        x=input_img_noised, filters=generator_power, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, strides=(1, 1),
    )
    d0_2 = residual_block(
        x=d0_1, filters=generator_power, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=False,
    )
    # Downsampling
    d1_1 = residual_block(
        x=d0_2, filters=generator_power * 2, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=True, use_pooling_layer=True
    )
    d1_2 = residual_block(
        x=d1_1, filters=generator_power * 2, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=False,
    )
    d2_1 = residual_block(
        x=d1_2, filters=generator_power * 4, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=True, use_pooling_layer=True
    )
    d2_2 = residual_block(
        x=d2_1, filters=generator_power * 4, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=False,
    )
    d3_1 = residual_block(
        x=d2_2, filters=generator_power * 8, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=True, use_pooling_layer=False
    )
    d3_2 = residual_block(
        x=d3_1, filters=generator_power * 8, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=False
    )

    # Upsampling
    u3_2 = deconv2d(d3_2, d3_1, generator_power * 8, kernel_initializer=kernel_initializer,
                    upsample=False,)
    u3_1 = deconv2d(u3_2, d2_2, generator_power * 4, kernel_initializer=kernel_initializer,
                    upsample=True, use_upsampling_layer=True)
    u2_2 = deconv2d(u3_1, d2_1, generator_power * 4, kernel_initializer=kernel_initializer,
                    upsample=False,)
    u2_1 = deconv2d(u2_2, d1_2, generator_power * 2, kernel_initializer=kernel_initializer,
                    upsample=True, use_upsampling_layer=True)
    u1_2 = deconv2d(u2_1, d1_1, generator_power * 2, kernel_initializer=kernel_initializer,
                    upsample=False,)
    u1_1 = deconv2d(u1_2, d0_2, generator_power, kernel_initializer=kernel_initializer,
                    upsample=True, use_upsampling_layer=False)
    u0_2 = deconv2d(u1_1, d0_1, generator_power, kernel_initializer=kernel_initializer,
                    upsample=False,)
    u0_1 = residual_block_last(
        x=u0_2, filters=output_channels, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer, weight_decay=1e-4, downsample=False,
    )

    return Model(input_img, u0_1)


def build_discriminator(input_img_shape, output_img_shape, discriminator_power=32, kernel_initializer=DEFAULT_INITIALIZER):

    # this model output range [-1, 1]. control by ResidualLastBlock's sigmiod activation

    original_img = Input(shape=input_img_shape)
    man_or_model_mad_img = Input(shape=output_img_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([original_img, man_or_model_mad_img])

    d = conv2d_bn(
        x=combined_imgs, filters=discriminator_power, kernel_size=(3, 3), kernel_initializer=kernel_initializer,
        weight_decay=1e-4, strides=(1, 1),
    )

    d1 = residual_block(
        x=d, filters=discriminator_power, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=True, use_pooling_layer=True
    )
    d2 = residual_block(
        x=d1, filters=discriminator_power * 2, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
        weight_decay=1e-4, downsample=True, use_pooling_layer=True
    )
    # d3 = residual_block(x = d2, filters = discriminator_power*4, kernel_size = KERNEL_SIZE, kernel_initializer = init, weight_decay = 1e-4, downsample = True)

    validity = residual_block_last(x=d2, filters=1, kernel_size=KERNEL_SIZE, kernel_initializer=kernel_initializer,
                                   weight_decay=1e-4, downsample=False,)

    return Model([original_img, man_or_model_mad_img], validity)
