# tensorflow Module
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

# user Module
from .layers import conv2d_bn, residual_block, residual_block_last, deconv2d

DEFAULT_INITIALIZER = RandomNormal(mean=0.0, stddev=0.02)
KERNEL_SIZE = (3, 3)
WEIGHT_DECAY = 1e-2


def build_generator(
    input_img_shape,
    output_channels=4,
    depth=None,
    generator_power=32,
    kernel_initializer=DEFAULT_INITIALIZER,
):
    """U-Net Generator"""
    # this model output range [-1, 1]. control by ResidualLastBlock's sigmiod activation

    # Image input
    input_img = Input(shape=input_img_shape)
    input_img_noised = GaussianNoise(0.1)(input_img)

    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 3

    down_sample_layers = []

    fix_shape_layer_1 = conv2d_bn(
        x=input_img_noised,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        strides=(1, 1),
    )
    fix_shape_layer_2 = residual_block(
        x=fix_shape_layer_1,
        filters=generator_power,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=1e-4,
        downsample=False,
    )
    down_sample_layers.append((fix_shape_layer_1, fix_shape_layer_2))
    for depth_step in range(depth):
        # Downsampling
        down_sample_layer = residual_block(
            x=fix_shape_layer_2,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=True,
            use_pooling_layer=True,
        )
        print(f"{depth_step}:{down_sample_layer.shape}")
        fix_shape_layer_1 = residual_block(
            x=down_sample_layer,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=False,
        )
        print(f"{depth_step}:{fix_shape_layer_1.shape}")
        fix_shape_layer_2 = residual_block(
            x=fix_shape_layer_1,
            filters=generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=1e-4,
            downsample=False,
        )
        print(f"{depth_step}:{fix_shape_layer_2.shape}")
        layer_collection = (down_sample_layer, fix_shape_layer_1, fix_shape_layer_2)
        down_sample_layers.append(layer_collection)
    for depth_step in range(depth, 0, -1):
        print(depth_step)
        print(f"{depth_step}:{down_sample_layers[depth_step][2].shape}")
        print(f"{depth_step}:{down_sample_layers[depth_step][1].shape}")
        print(f"{depth_step}:{down_sample_layers[depth_step][0].shape}")
        if depth_step == depth:
            fix_shape_layer_1 = deconv2d(
                down_sample_layers[depth_step][2],
                down_sample_layers[depth_step][1],
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                kernel_initializer=kernel_initializer,
                upsample=False,
            )
        else:
            fix_shape_layer_1 = deconv2d(
                upsampling_layer,
                down_sample_layers[depth_step][1],
                generator_power * (2 ** depth_step),
                kernel_size=KERNEL_SIZE,
                kernel_initializer=kernel_initializer,
                upsample=False,
            )
        print(f"{depth_step}:{fix_shape_layer_1.shape}")
        fix_shape_layer_2 = deconv2d(
            fix_shape_layer_1,
            down_sample_layers[depth_step][2],
            generator_power * (2 ** depth_step),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            upsample=False,
        )
        print(f"{depth_step}:{fix_shape_layer_2.shape}")
        upsampling_layer = deconv2d(
            fix_shape_layer_2,
            down_sample_layers[depth_step - 1][0],
            generator_power * (2 ** (depth_step - 1)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            upsample=True,
            use_upsampling_layer=False,
        )
        print(f"{depth_step}:{upsampling_layer.shape}")
    output_layer = residual_block_last(
        x=upsampling_layer,
        filters=output_channels,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
    )
    print(output_layer.shape)
    return Model(input_img, output_layer)


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
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    validity = residual_block_last(
        x=down_sampled_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
    )

    return Model([original_img, man_or_model_mad_img], validity)


def build_classifier(
    input_img_shape,
    classfier_power=32,
    depth=None,
    num_class=2,
    kernel_initializer=DEFAULT_INITIALIZER,
):

    # this model output range [0, 1]. control by ResidualLastBlock's sigmiod activation

    input_img = Input(shape=input_img_shape)
    dense_unit = input_img_shape[0]
    if depth is None:
        img_size = input_img_shape[0]
        depth = 0
        while img_size != 1:
            img_size //= 2
            depth += 1
        depth -= 5
    down_sampled_layer = conv2d_bn(
        x=input_img,
        filters=classfier_power,
        kernel_size=(3, 3),
        kernel_initializer=kernel_initializer,
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=classfier_power * (2 ** (depth_step // 4)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=classfier_power * (2 ** (depth_step // 4)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        down_sampled_layer = residual_block(
            x=down_sampled_layer,
            filters=classfier_power * (2 ** (depth_step // 4)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )
    # (BATCH_SIZE, 32, 32, Filters)
    output = GlobalMaxPooling2D()(down_sampled_layer)
    output = Dense(units=dense_unit)(output)
    output = BatchNormalization(axis=-1)(output)
    # (BATCH_SIZE, 1024)
    output = Dense(units=num_class, activation="sigmoid", kernel_initializer="he_normal")(output)
    # (BATCH_SIZE, NUM_CLASS)
    return Model(input_img, outputs=output)


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
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        filter_growing_layer = residual_block(
            x=filter_growing_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        filter_growing_layer = residual_block(
            x=filter_growing_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    filter_growing_validity = residual_block_last(
        x=filter_growing_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=WEIGHT_DECAY,
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
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth - 1, -1, -1):
        filter_shrinking_layer = residual_block(
            x=filter_shrinking_layer,
            filters=discriminator_power * (2 ** ((depth_step + 2) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        filter_shrinking_layer = residual_block(
            x=filter_shrinking_layer,
            filters=discriminator_power * (2 ** ((depth_step + 1) // 2)),
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    filter_shrinking_validity = residual_block_last(
        x=filter_shrinking_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=WEIGHT_DECAY,
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
        weight_decay=WEIGHT_DECAY,
        strides=1,
    )
    for depth_step in range(depth):
        filter_fixed_layer = residual_block(
            x=filter_fixed_layer,
            filters=discriminator_power,
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=False,
        )
        filter_fixed_layer = residual_block(
            x=filter_fixed_layer,
            filters=discriminator_power,
            kernel_size=KERNEL_SIZE,
            kernel_initializer=kernel_initializer,
            weight_decay=WEIGHT_DECAY,
            downsample=True,
            use_pooling_layer=False,
        )

    filter_fixed_validity = residual_block_last(
        x=filter_fixed_layer,
        filters=1,
        kernel_size=KERNEL_SIZE,
        kernel_initializer=kernel_initializer,
        weight_decay=WEIGHT_DECAY,
        downsample=False,
    )
    validity = layers.Average()(
        [filter_growing_validity, filter_shrinking_validity, filter_fixed_validity]
    )
    return Model([original_img, man_or_model_mad_img], validity)
