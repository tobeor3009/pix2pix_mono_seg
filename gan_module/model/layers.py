# this model avgpool(pool_size=2, stride=2)-conv(kernel_size=1, stride=1)
# when down_sampled residual block

from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


default_initializer = RandomNormal(mean=0.0, stddev=0.02)
NEGATIVE_RATIO = 0.25
SIGMOID_NEGATIVE_RATIO = 0.25
# in batchnomalization, gamma_initializer default is "ones", but if use with residual shortcut,
# set to 0 this value will helpful for Early training
GAMMA_INITIALIZER = "zeros"


def conv2d_bn(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    strides=(1, 1),
    use_pooling_layer=False
):
    if use_pooling_layer:
        cnn_strides = (1, 1)
        pooling_stride = strides
    else:
        cnn_strides = strides

    layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=cnn_strides,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer=kernel_initializer,
    )(x)
    layer = BatchNormalization(axis=-1)(layer)
    layer = LeakyReLU(NEGATIVE_RATIO)(layer)
    if use_pooling_layer:
        layer = MaxPooling2D(pool_size=pooling_stride,
                             strides=pooling_stride, padding="same")(layer)
    return layer


def residual_block(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    downsample=True,
    use_pooling_layer=False
):
    if downsample:
        stride = 2
        residual = AveragePooling2D(
            pool_size=stride, strides=stride, padding="same")(x)
        residual = conv2d_bn(
            residual, filters, kernel_size=1, kernel_initializer=kernel_initializer, strides=1
        )
    else:
        residual = x
        stride = 1

    conved = conv2d_bn(
        x,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=stride,
        use_pooling_layer=use_pooling_layer
    )
    conved = conv2d_bn(
        conved,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=1
    )
    output = layers.add([conved, residual])
    output = BatchNormalization(
        axis=-1, gamma_initializer=GAMMA_INITIALIZER)(output)
    output = LeakyReLU(NEGATIVE_RATIO)(output)

    return output


def wide_residual_block(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    width=4,
    downsample=True,
    use_pooling_layer=False,

):
    if downsample:
        stride = 2
        residual = AveragePooling2D(
            pool_size=stride, strides=stride, padding="same")(x)
        residual = conv2d_bn(
            residual, filters, kernel_size=1, kernel_initializer=kernel_initializer, strides=1
        )
    else:
        residual = x
        stride = 1
    conved_stacked = []
    for _ in range(width):
        conved = conv2d_bn(
            x,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            weight_decay=weight_decay,
            strides=stride,
            use_pooling_layer=use_pooling_layer
        )
        conved = conv2d_bn(
            conved,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            weight_decay=weight_decay,
            strides=1
        )
        added_layer = layers.add([conved, residual])
        added_layer = BatchNormalization(
            axis=-1, gamma_initializer=GAMMA_INITIALIZER)(added_layer)
        added_layer = LeakyReLU(NEGATIVE_RATIO)(added_layer)
        conved_stacked.append(added_layer)

    conved_stacked = layers.concatenate(conved_stacked)
    conved_stacked = BatchNormalization(axis=-1)(conved_stacked)
    conved_stacked = LeakyReLU(NEGATIVE_RATIO)(conved_stacked)

    return conved_stacked


def residual_block_last(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    downsample=True,
    activation='sigmoid'
):
    if downsample:
        stride = 2
        residual = AveragePooling2D(pool_size=stride)(x)
        residual = conv2d_bn(
            residual, filters, kernel_size=1, kernel_initializer=kernel_initializer, strides=1
        )
    else:
        residual = conv2d_bn(
            x, filters, kernel_size=1, kernel_initializer=kernel_initializer, strides=1
        )
        stride = 1

    conved = conv2d_bn(
        x,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=stride,
    )
    conved = conv2d_bn(
        conved,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=1,
    )
    output = layers.add([conved, residual])
    output = BatchNormalization(
        axis=-1, gamma_initializer=GAMMA_INITIALIZER)(output)
    output = LeakyReLU(SIGMOID_NEGATIVE_RATIO)(output)
    # use if you want output range [0,1]
    if activation == 'sigmoid':
        output = sigmoid(output)
    # use if you want output range [-1,1]
    elif activation == 'tanh':
        output = tanh(output)
    return output


def deconv2d(
    layer_input,
    skip_input,
    filters,
    f_size=4,
    kernel_initializer=default_initializer,
    upsample=True,
    use_upsampling_layer=False


):
    """Layers used during upsampling"""

    strides = 2 if upsample else 1
    if use_upsampling_layer:
        layer_input = Conv2DTranspose(
            filters,
            kernel_size=f_size,
            strides=1,
            padding="same",
            kernel_regularizer=l2(0.0),
            kernel_initializer=kernel_initializer
        )(layer_input)
        layer_input = UpSampling2D(
            size=strides, interpolation='nearest')(layer_input)
    else:
        layer_input = Conv2DTranspose(
            filters,
            kernel_size=f_size,
            strides=strides,
            padding="same",
            kernel_regularizer=l2(0.0),
            kernel_initializer=kernel_initializer
        )(layer_input)
    layer_input = Concatenate()([layer_input, skip_input])
    layer_input = BatchNormalization(axis=-1)(layer_input)
    layer_input = LeakyReLU(NEGATIVE_RATIO)(layer_input)
    return layer_input
