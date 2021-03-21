# tensorflow Module
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GaussianNoise
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

# user Module
from .layers import conv2d_bn, residual_block, residual_block_last, deconv2d

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
        x=input_img_noised,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        strides=(1, 1),
    )
    d0_2 = residual_block(
        x=d0_1,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    # Downsampling
    d1_1 = residual_block(
        x=d0_2,
        filters=generator_power * 2,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=True,
        use_pooling_layer=True,
    )
    d1_2 = residual_block(
        x=d1_1,
        filters=generator_power * 2,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    d2_1 = residual_block(
        x=d1_2,
        filters=generator_power * 4,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=True,
        use_pooling_layer=True,
    )
    d2_2 = residual_block(
        x=d2_1,
        filters=generator_power * 4,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    d3_1 = residual_block(
        x=d2_2,
        filters=generator_power * 8,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=True,
        use_pooling_layer=False,
    )
    d3_2 = residual_block(
        x=d3_1,
        filters=generator_power * 8,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    d4_1 = residual_block(
        x=d3_2,
        filters=generator_power * 16,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=True,
        use_pooling_layer=False,
    )
    d4_2 = residual_block(
        x=d4_1,
        filters=generator_power * 16,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    d5_1 = residual_block(
        x=d4_2,
        filters=generator_power * 32,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=True,
        use_pooling_layer=False,
    )
    d5_2 = residual_block(
        x=d5_1,
        filters=generator_power * 32,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    d6_1 = residual_block(
        x=d5_2,
        filters=generator_power * 64,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=True,
        use_pooling_layer=False,
    )
    d6_2 = residual_block(
        x=d6_1,
        filters=generator_power * 64,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    # Upsampling
    u6_2 = deconv2d(
        d6_2,
        d6_1,
        generator_power * 64,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u6_1 = deconv2d(
        u6_2,
        d5_2,
        generator_power * 32,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=True,
        use_upsampling_layer=True,
    )
    u5_2 = deconv2d(
        u6_1,
        d5_1,
        generator_power * 32,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u5_1 = deconv2d(
        u5_2,
        d4_2,
        generator_power * 16,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=True,
        use_upsampling_layer=True,
    )
    u4_2 = deconv2d(
        u5_1,
        d4_1,
        generator_power * 16,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u4_1 = deconv2d(
        u4_2,
        d3_2,
        generator_power * 8,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=True,
        use_upsampling_layer=True,
    )
    u3_2 = deconv2d(
        u4_1,
        d3_1,
        generator_power * 8,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u3_1 = deconv2d(
        u3_2,
        d2_2,
        generator_power * 4,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=True,
        use_upsampling_layer=True,
    )
    u2_2 = deconv2d(
        u3_1,
        d2_1,
        generator_power * 4,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u2_1 = deconv2d(
        u2_2,
        d1_2,
        generator_power * 2,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=True,
        use_upsampling_layer=True,
    )
    u1_2 = deconv2d(
        u2_1,
        d1_1,
        generator_power * 2,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u1_1 = deconv2d(
        u1_2,
        d0_2,
        generator_power,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=True,
        use_upsampling_layer=False,
    )
    u0_2 = deconv2d(
        u1_1,
        d0_1,
        generator_power,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        upsample=False,
    )
    u0_1 = residual_block_last(
        x=u0_2,
        filters=output_channels,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    return Model(input_img, u0_1)


def build_discriminator(
    input_img_shape,
    output_img_shape,
    depth=None,
    discriminator_power=32,
    kernel_initializer=DEFAULT_INITIALIZER,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_img = Input(shape=input_img_shape)
    man_or_model_mad_img = Input(shape=output_img_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([original_img, man_or_model_mad_img])

    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 3
    down_sampled_layer = conv2d_bn(
        x=combined_imgs,
        filters=discriminator_power,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        strides=1,
    )
    for depth_step in range(depth):
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=False,
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
        )

    validity = residual_block_last(
        x=down_sampled_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )

    return Model([original_img, man_or_model_mad_img], validity)


def build_ensemble_discriminator(
    input_img_shape,
    output_img_shape,
    depth=None,
    discriminator_power=32,
    kernel_initializer=DEFAULT_INITIALIZER,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    original_img = Input(shape=input_img_shape)
    man_or_model_mad_img = Input(shape=output_img_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([original_img, man_or_model_mad_img])

    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 3

    # ----------------------------
    #  Define Filter Growing Layer
    # ----------------------------
    filter_growing_layer = conv2d_bn(
        x=combined_imgs,
        filters=discriminator_power,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        strides=1,
    )
    for depth_step in range(depth):
        filter_growing_layer = residual_block(
            x=filter_growing_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=False,
        )
        filter_growing_layer = residual_block(
            x=filter_growing_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
        )

    filter_growing_validity = residual_block_last(
        x=filter_growing_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    # ----------------------------
    #  Define Filter Shrinking Layer
    # ----------------------------
    filter_shrinking_layer = conv2d_bn(
        x=combined_imgs,
        filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        strides=1,
    )
    for depth_step in range(depth - 1, -1, -1):
        filter_shrinking_layer = residual_block(
            x=filter_shrinking_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=False,
        )
        filter_shrinking_layer = residual_block(
            x=filter_shrinking_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
        )

    filter_shrinking_validity = residual_block_last(
        x=filter_shrinking_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    # ----------------------------
    #  Define Filter Fixed Layer
    # ----------------------------
    filter_fixed_layer = conv2d_bn(
        x=combined_imgs,
        filters=discriminator_power,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        strides=1,
    )
    for depth_step in range(depth):
        filter_fixed_layer = residual_block(
            x=filter_fixed_layer,
            filters=discriminator_power,
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=False,
        )
        filter_fixed_layer = residual_block(
            x=filter_fixed_layer,
            filters=discriminator_power,
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
        )

    filter_fixed_validity = residual_block_last(
        x=filter_fixed_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    validity = layers.Average()(
        [filter_growing_validity, filter_shrinking_validity, filter_fixed_validity]
    )
    return Model([original_img, man_or_model_mad_img], validity)
