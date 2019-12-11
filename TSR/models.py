from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, DepthwiseConv2D
from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape, ReLU

from tensorflow.keras import backend as K


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    :param inputs: Tensor, input tensor of conv layer.
    :param filters: Integer, the dimensionality of the output space.
    :param kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
    :param strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    :return: Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return ReLU(6.)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    :param inputs: Tensor, input tensor of conv layer.
    :param filters: Integer, the dimensionality of the output space.
    :param kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
    :param t: Integer, expansion factor.
            t is always applied to the input size.
    :param s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
    :param r: Boolean, Whether to use the residuals.
    :return: Output tensor.
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
    :param inputs: Tensor, input tensor of conv layer.
    :param filters: Integer, the dimensionality of the output space.
    :param kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
    :param t: Integer, expansion factor.
            t is always applied to the input size.
    :param strides: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
    :param n: Integer, layer repeat times.
    :return: Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, s=strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, s=1, r=True)

    return x


def mobilenet_v2(input_shape, k):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    :param input_shape: An integer or tuple/list of 3 integers, shape of input tensor.
    :param k: Integer, number of classes
    :return: TruncatedMobileNetv2 model.
    """
    fc_count = 320

    inputs = Input(shape=input_shape, name='input_img')
    x = inputs

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

    x = Reshape((k,), name='out_reshape')(x)
    x = Activation('softmax', name='softmax')(x)

    output = x
    model = Model(inputs, output)

    return model


def create(
        base_name,
        num_classes,
        input_shape=(32, 32, 3)
):
    """
    Create model with
    :param base_name: Name of the base of the network
    :param num_classes: An integer or tuple/list of 3 integers, shape of input tensor.
    :param input_shape: Integer, number of classes
    :return: Keras model
    """
    base_list = { "TSR_CstmMobileNetv2": mobilenet_v2 }

    if base_name not in base_list:
        print('No such model {}'.format(base_name))
        return None

    return base_list[base_name](input_shape, num_classes)

# Simple testing
if __name__ == '__main__':
    create(10, (32, 32, 3))
