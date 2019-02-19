from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, DepthwiseConv2D, MaxPooling2D, UpSampling2D, Deconv2D
from keras.layers import Activation, BatchNormalization, add, Reshape, ReLU, concatenate
from keras import backend as K
import tensorflow as tf

def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return ReLU(6.)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
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
    x = _bottleneck(inputs, filters, kernel, t, s=strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, s=1, r=True)

    return x


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def intersect_over_union(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    return K.switch(K.equal(union, 0), K.variable(1), intersection / union)


def iou_loss(y_true, y_pred):
    return 1 - intersect_over_union(y_true, y_pred)


def pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
    return - tf.reduce_sum(target * tf.log(output))


def result_loss(y_true, y_pred):
    # return pixelwise_crossentropy(y_true, y_pred) / 10 + iou_loss(y_true, y_pred)
    return iou_loss(y_true, y_pred)


def unet(input_shape):
    _input = Input(input_shape, name='input_img')
    x = _input

    x2 = _conv_block(x, 8, 3, strides=2)
    x4 = _conv_block(x2, 16, 3, strides=2)
    x = _inverted_residual_block(x4, 24, (3, 3), t=1, strides=1, n=1)
    x8 = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=2)
    x16 = _inverted_residual_block(x8, 48, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x16, 64, (3, 3), t=6, strides=1, n=1)

    x = Deconv2D(32, kernel_size=2, strides=2)(x)
    x = add([x8, x])
    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(6.)(x)

    x = Deconv2D(16, kernel_size=2, strides=2)(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(6.)(x)
    x = add([x4, x])
    x = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(6.)(x)

    x = Deconv2D(8, kernel_size=2, strides=2)(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(6.)(x)
    x = add([x2, x])
    x = Conv2D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(6.)(x)

    # conv1 = Conv2D(int(32 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(_input)
    # x = Conv2D(int(32 * a), 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')(conv1)
    #
    # conv2 = Conv2D(int(128 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # x = Conv2D(int(128 * a), 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')(conv2)
    #
    # conv3 = Conv2D(int(256 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # x = Conv2D(int(256 * a), 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')(conv3)
    #
    # # conv4 = Conv2D(int(512 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = Conv2D(int(512 * a), 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')(conv4)
    #
    # x = Conv2D(int(512 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = Conv2D(int(1024 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    #
    # x = Deconv2D(int(256 * a), kernel_size=2, strides=2)(x)
    # # x = Conv2D(int(256 * a), 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # x = add([conv3, x])
    # x = Conv2D(int(256 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = Conv2D(int(512 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    #
    # x = Deconv2D(int(128 * a), kernel_size=2, strides=2)(x)
    # # x = Conv2D(int(128 * a), 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # x = add([conv2, x])
    # x = Conv2D(int(128 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = Conv2D(int(256 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    #
    # # x = Deconv2D(int(128 * a), kernel_size=2, strides=2)(x)
    # # x = Conv2D(int(128 * a), 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = concatenate([conv2, x], axis=3)
    # # x = Conv2D(int(128 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = Conv2D(int(128 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    #
    # # x = Deconv2D(int(64 * a), kernel_size=2, strides=2)(x)
    # # x = Conv2D(int(64 * a), 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = concatenate([conv1, x], axis=3)
    # # x = Conv2D(int(64 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # # x = Conv2D(int(64 * a), 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # x = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(1, 1, activation='sigmoid')(x)

    _output = x
    model = Model(_input, _output)

    return model


def create(
        input_shape,
        base=None
):
    return unet(input_shape)


if __name__ == '__main__':
    from keras.utils.layer_utils import print_summary

    net = create(input_shape=(240, 320, 3))
    print_summary(net)
