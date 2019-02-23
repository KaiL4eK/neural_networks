from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, DepthwiseConv2D
from keras.layers import Activation, BatchNormalization, add, Reshape, ReLU
from keras import backend as K


def _make_divisible(v, divisor, min_value=None):
    divisor = int(v)
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def _bottleneck(inputs, filters, kernel, t, s, r, block_id, alpha=1.0):
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
    prefix = 'block_{}_'.format(block_id)

    x = inputs

    if block_id:
        x = Conv2D(tchannel, 1,
                   padding='same',
                   strides=1,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                               epsilon=1e-3,
                               momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix='expanded_conv_'

    x = DepthwiseConv2D(kernel, strides=(s, s),
                                depth_multiplier=1,
                                padding='same',
                                use_bias=False,
                                name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    pointwise_filters = _make_divisible(filters * alpha, 8)

    x = Conv2D(pointwise_filters, 1,
               strides=1,
               padding='same',
               name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                           momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n, block_id, alpha=1.0):
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

    x = _bottleneck(inputs, filters, kernel, t, strides, r=False, block_id=block_id, alpha=alpha)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, r=True, block_id=block_id + i, alpha=alpha)

    return x
