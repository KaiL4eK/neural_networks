from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary

import numpy as np
import cv2
import config

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

nn_img_side = 240
nn_out_size = 120


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/51553
def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    return (intersection + smooth) / (union + smooth)


def intersect_over_union(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    # return K.switch( K.equal(union, 0), K.variable(1), intersection / union) 

    return (intersection + 1) / (union + 1)


def iou_loss(y_true, y_pred):
    return 1 - intersect_over_union(y_true, y_pred)


def iou_metrics(y_true, y_pred):
    return intersect_over_union(y_true, y_pred)


def bin_ce_n_iou(y_true, y_pred):
    return iou_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)


### Image preprocessing ###

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
    img = cv2.resize(img, (nn_img_side, nn_img_side), interpolation=cv2.INTER_CUBIC)

    # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = img.astype('float32', copy=False)
    # img[:,:,0] -= 103.939
    # img[:,:,1] -= 116.779
    # img[:,:,2] -= 123.68
    img /= 255.

    return img


def preprocess_mask(img):
    img = cv2.resize(img, (nn_out_size, nn_out_size), interpolation=cv2.INTER_NEAREST)
    img = img.astype('float32', copy=False)
    img /= 255.

    return img


####################################

def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _deconv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), bn_epsilon=1e-3,
                  bn_momentum=0.99, block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = filters * alpha
    filters = _make_divisible(filters)
    x = Conv2DTranspose(filters, kernel,
                        padding='valid',
                        use_bias=False,
                        strides=strides,
                        name='deconv%d' % block_id,
                        kernel_initializer='glorot_normal')(inputs)

    return Activation(relu6, name='deconv%d_relu' % block_id)(x)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), bn_epsilon=1e-3,
                bn_momentum=0.99, block_id=1):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        bn_epsilon: Epsilon value for BatchNormalization
        bn_momentum: Momentum value for BatchNormalization
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = filters * alpha
    filters = _make_divisible(filters)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv%d' % block_id,
               kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv%d_bn' % block_id)(x)
    return Activation(relu6, name='conv%d_relu' % block_id)(x)


def _depthwise_conv_block_v2(inputs, pointwise_conv_filters, alpha, expansion_factor,
                             depth_multiplier=1, strides=(1, 1), bn_epsilon=1e-3,
                             bn_momentum=0.99, block_id=1):
    """Adds a depthwise convolution block V2.
    A depthwise convolution V2 block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        expansion_factor: controls the expansion of the internal bottleneck
            blocks. Should be a positive integer >= 1
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        bn_epsilon: Epsilon value for BatchNormalization
        bn_momentum: Momentum value for BatchNormalization
        block_id: Integer, a unique identification designating the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    depthwise_conv_filters = _make_divisible(input_shape[channel_axis] * expansion_factor)
    pointwise_conv_filters = _make_divisible(pointwise_conv_filters * alpha)

    if depthwise_conv_filters > input_shape[channel_axis]:
        x = Conv2D(depthwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   name='conv_expand_%d' % block_id,
                   kernel_initializer='glorot_normal')(inputs)

        x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                               name='conv_expand_%d_bn' % block_id)(x)

        x = Activation(relu6, name='conv_expand_%d_relu' % block_id)(x)

    else:
        x = inputs

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id,
                        kernel_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id,
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv_pw_%d_bn' % block_id)(x)

    if strides == (2, 2):
        return x
    else:
        if input_shape[channel_axis] == pointwise_conv_filters:
            x = add([inputs, x])

    return x


global_blk_idx = 0


def get_block_idx():
    global global_blk_idx
    global_blk_idx += 1
    return global_blk_idx


def mobilenetv2(input_shape, lr, alpha=0.5, expansion_factor=6, depth_multiplier=1):
    input = Input(input_shape, name=config.INPUT_TENSOR_NAMES[0])

    x = _conv_block(input, 32, alpha, bn_epsilon=1e-3, strides=(2, 2))
    x_d2 = x

    x = _depthwise_conv_block_v2(x, 16, alpha, 1, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, block_id=get_block_idx(),
                                 bn_epsilon=1e-3, bn_momentum=0.999, strides=(2, 2))
    x_d4 = x

    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=get_block_idx(),
                                 bn_epsilon=1e-3, bn_momentum=0.999, strides=(2, 2))
    x_d8 = x

    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())
    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=get_block_idx(),
                                 bn_epsilon=1e-3, bn_momentum=0.999, strides=(2, 2))
    x_d16 = x

    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _depthwise_conv_block_v2(x, 96, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())
    x = _depthwise_conv_block_v2(x, 96, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())
    x = _depthwise_conv_block_v2(x, 96, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _depthwise_conv_block_v2(x, 160, alpha, expansion_factor, depth_multiplier, block_id=get_block_idx(),
                                 bn_epsilon=1e-3, bn_momentum=0.999, strides=(2, 2))
    x_d32 = x

    x = _depthwise_conv_block_v2(x, 160, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())
    x = _depthwise_conv_block_v2(x, 160, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _depthwise_conv_block_v2(x, 320, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    ################

    x = _deconv_block(x, 64, alpha, kernel=2, strides=2, bn_epsilon=1e-3, bn_momentum=0.999, block_id=get_block_idx())
    x = Add()([x, x_d16])

    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _deconv_block(x, 32, alpha, kernel=2, strides=2, bn_epsilon=1e-3, bn_momentum=0.999, block_id=get_block_idx())
    x = Add()([x, x_d8])

    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _deconv_block(x, 24, alpha, kernel=2, strides=2, bn_epsilon=1e-3, bn_momentum=0.999, block_id=get_block_idx())
    x = Add()([x, x_d4])

    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
                                 block_id=get_block_idx())

    x = _deconv_block(x, 16, alpha, kernel=2, strides=2, bn_epsilon=1e-3, bn_momentum=0.999, block_id=get_block_idx())
    # x = Add()([x, x_d2])

    # x = _depthwise_conv_block_v2(x, 8, alpha, expansion_factor, depth_multiplier, bn_epsilon=1e-3, bn_momentum=0.999,
    #                              block_id=get_block_idx())

    x = _deconv_block(x, 8, alpha, kernel=2, strides=2, bn_epsilon=1e-3, bn_momentum=0.999, block_id=get_block_idx())

    output_lin = Conv2D(filters=1, kernel_size=1, padding='same', use_bias=False, name=config.OUTPUT_TENSOR_NAMES[0],
                        kernel_initializer='glorot_normal')(x)
    output_sig = Activation(activation='sigmoid')(output_lin)

    train_model = Model(input, output_sig)
    infer_model = Model(input, output_lin)

    train_model.compile(optimizer=Adam(lr=lr), loss=bin_ce_n_iou, metrics=['accuracy', iou_metrics])

    return train_model, infer_model


def create_model(input_shape=(160, 320, 3), lr=1e-3):
    # return get_unet(input_shape=input_shape, lr=lr)
    # return get_unet_simple(input_shape=input_shape, lr=lr)
    return mobilenetv2(input_shape=input_shape, lr=lr)
