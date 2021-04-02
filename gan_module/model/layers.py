# this model avgpool(pool_size=2, stride=2)-conv(kernel_size=1, stride=1)
# when down_sampled residual block

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


default_initializer = RandomNormal(mean=0.0, stddev=0.02)
NEGATIVE_RATIO = 0.25
# in batchnomalization, gamma_initializer default is "ones", but if use with residual shortcut,
# set to 0 this value will helpful for Early training
GAMMA_INITIALIZER = "zeros"
HIGHWAY_INIT_BIAS = -3


def conv2d_bn(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    strides=(1, 1),
    use_pooling_layer=False,
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
        # Batch Normalization Beta replace Conv2D Bias
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer=kernel_initializer,
    )(x)
    layer = BatchNormalization(axis=-1)(layer)
    layer = LeakyReLU(NEGATIVE_RATIO)(layer)
    if use_pooling_layer:
        layer = AveragePooling2D(pool_size=pooling_stride, strides=pooling_stride, padding="same")(
            layer
        )
    return layer


def residual_block(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    downsample=True,
    use_pooling_layer=False,
    Last=False,
):
    if downsample:
        stride = 2
        residual = AveragePooling2D(pool_size=stride, strides=stride, padding="same")(x)
        residual = conv2d_bn(
            residual, filters, kernel_size=1, kernel_initializer=kernel_initializer, strides=1
        )
        transform_gate_output = transform_gate(residual, filters)
    else:
        stride = 1
        if Last:
            residual = conv2d_bn(
                x=x,
                filters=filters,
                kernel_size=1,
                kernel_initializer=kernel_initializer,
                strides=1,
            )
        else:
            residual = x
        transform_gate_output = transform_gate(residual, filters)

    conved = conv2d_bn(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=stride,
        use_pooling_layer=use_pooling_layer,
    )
    conved = conv2d_bn(
        x=conved,
        filters=filters,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        strides=1,
    )
    output = layers.add([conved, residual])
    output = BatchNormalization(axis=-1, gamma_initializer=GAMMA_INITIALIZER)(output)
    output = LeakyReLU(0.25)(output)
    return output * transform_gate_output + residual * (1 - transform_gate_output)


def residual_block_last(
    x,
    filters,
    kernel_size,
    kernel_initializer=default_initializer,
    weight_decay=0.0,
    downsample=True,
    activation="sigmoid",
):
    output = residual_block(
        x, filters, kernel_size, kernel_initializer, weight_decay, downsample, Last=True
    )
    # use if you want output range [0,1]
    if activation == "sigmoid":
        output = sigmoid(output)
    # use if you want output range [-1,1]
    elif activation == "tanh":
        output = tanh(output)
    return output


def deconv2d(
    layer_input,
    skip_input,
    filters,
    kernel_size=3,
    kernel_initializer=default_initializer,
    upsample=True,
    use_upsampling_layer=False,
):
    """Layers used during upsampling"""

    strides = 2 if upsample else 1
    skip_input_channel = skip_input.shape[-1]
    if use_upsampling_layer:
        layer_input = UpSampling2D(size=strides, interpolation="nearest")(layer_input)
        layer_input = conv2d_bn(
            x=layer_input,
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            strides=1,
        )
    else:
        layer_input = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            # Batch Normalization Beta replace Conv2D Bias
            use_bias=False,
            kernel_regularizer=l2(0.0),
            kernel_initializer=kernel_initializer,
        )(layer_input)
    layer_input = BatchNormalization(axis=-1)(layer_input)
    concatenated = Concatenate()([layer_input, skip_input])
    transform_gate_output = transform_gate(concatenated, filters + skip_input_channel)
    output = conv2d_bn(
        x=concatenated,
        filters=filters + skip_input_channel,
        kernel_initializer=kernel_initializer,
        kernel_size=kernel_size,
        strides=1,
    )
    return output * transform_gate_output + concatenated * (1 - transform_gate_output)


def transform_gate(x, highway_dim):

    transform = Dense(
        units=highway_dim, bias_initializer=tf.constant_initializer(HIGHWAY_INIT_BIAS)
    )(x)
    transform = sigmoid(transform)
    return transform
