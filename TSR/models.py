from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, DepthwiseConv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, add, Reshape, ReLU

from keras import backend as K


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return ReLU(6.)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = ReLU(6.)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, s=strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, s=1, r=True)

    return x


def mobilenet_v2(input_shape, k):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """
    fc_count = 320

    inputs = Input(shape=input_shape, name='input_img')
    x = inputs

    # New variant
    x = _conv_block(x, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, fc_count, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, fc_count))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same')(x)

    # x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    # x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    # x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    # x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    # x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    # x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    # x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    # x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    # fcCount = 1280
    # x = _conv_block(x, fcCount, (1, 1), strides=(1, 1))
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, fcCount))(x)
    # x = Dropout(0.3, name='Dropout')(x)
    # x = Conv2D(k, (1, 1), padding='same')(x)

    x = Reshape((k,), name='out_reshape')(x)
    x = Activation('softmax', name='softmax')(x)

    output = x
    model = Model(inputs, output)

    return model


def create(
        base,
        num_classes,
        input_shape=(32, 32, 3)
):
    return mobilenet_v2(input_shape, num_classes)


if __name__ == '__main__':
    create(10, (32, 32, 3))