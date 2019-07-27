from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU
from keras.layers import Concatenate, MaxPooling2D, UpSampling2D
from keras.regularizers import l2

from functools import wraps, reduce


class DetBackend(object):
    def __init__(self):
        self.pred_layers = []
        self.input_layers = []

        self.head_layers_cnt = 0
        self.downgrades = [1, 1]


class MobileNetV2(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']
        pred_filter_count = kwargs['pred_filters']

        image_input = Input(shape=train_shape, name='input_img')

        original_model = True
        ##########################
        import mobilenet_utils as mnu
        import os

        alpha = self.alpha

        if original_model:
            from keras.applications.mobilenetv2 import MobileNetV2
            mobilenetv2 = MobileNetV2(
                input_tensor=image_input, include_top=False, weights=None, alpha=alpha)
            x = mobilenetv2.output
        else:
            channel_axis = 1 if mnu.K.image_data_format() == 'channels_first' else -1

            first_block_filters = mnu._make_divisible(32 * alpha, 8)

            x = mnu.Conv2D(first_block_filters, 3, padding='same',
                           strides=2, use_bias=False, name='Conv1')(image_input)
            x = mnu.BatchNormalization(axis=channel_axis, name='bn_Conv1')(x)
            x = mnu.ReLU(6., name='Conv1_relu')(x)

            x = mnu._inverted_residual_block(
                x, 16, 3, t=1, strides=1, n=1, alpha=alpha, block_id=0)
            x = mnu._inverted_residual_block(
                x, 24, 3, t=6, strides=2, n=2, alpha=alpha, block_id=1)
            x = mnu._inverted_residual_block(
                x, 32, 3, t=6, strides=2, n=3, alpha=alpha, block_id=3)
            x = mnu._inverted_residual_block(
                x, 64, 3, t=6, strides=2, n=4, alpha=alpha, block_id=6)
            x = mnu._inverted_residual_block(
                x, 96, 3, t=6, strides=1, n=3, alpha=alpha, block_id=10)
            x = mnu._inverted_residual_block(
                x, 160, 3, t=6, strides=2, n=3, alpha=alpha, block_id=13)
            x = mnu._inverted_residual_block(
                x, 320, 3, t=6, strides=1, n=1, alpha=alpha, block_id=16)

            # last_block_filters = mnu._make_divisible(1280 * alpha, 8)
            last_block_filters = 1280
            x = mnu.Conv2D(last_block_filters, 1, padding='same',
                           strides=1, use_bias=False, name='Conv_1')(x)
            x = mnu.BatchNormalization(axis=channel_axis, name='Conv_1_bn')(x)
            x = mnu.ReLU(6., name='out_relu')(x)

        ##########################
        output = x
        pred_yolo_1 = Conv2D(pred_filter_count, 1, padding='same',
                             strides=1, name='DetectionLayer1')(output)

        self.pred_layers = [pred_yolo_1]
        self.input_layers = [image_input]

        self.head_layers_cnt = -1
        self.downgrades = [32, 16]


class MobileNetV2_35(MobileNetV2):
    def __init__(self, **kwargs):
        self.alpha = 0.35
        super().__init__(**kwargs)


class MobileNetV2_50(MobileNetV2):
    def __init__(self, **kwargs):
        self.alpha = 0.5
        super().__init__(**kwargs)


class MobileNetV2_75(MobileNetV2):
    def __init__(self, **kwargs):
        self.alpha = 0.75
        super().__init__(**kwargs)


class MobileNetV2_100(MobileNetV2):
    def __init__(self, **kwargs):
        self.alpha = 1
        super().__init__(**kwargs)


class Tiny_YOLOv3(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']
        pred_filter_count = kwargs['pred_filters']

        image_input = Input(shape=train_shape, name='input_img')

        @wraps(Conv2D)
        def DarknetConv2D(*args, **kwargs):
            """Wrapper to set Darknet parameters for Convolution2D."""
            darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
            darknet_conv_kwargs['padding'] = 'valid' if kwargs.get(
                'strides') == (2, 2) else 'same'
            darknet_conv_kwargs.update(kwargs)
            return Conv2D(*args, **darknet_conv_kwargs)

        def DarknetConv2D_BN_Leaky(*args, **kwargs):
            """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
            no_bias_kwargs = {'use_bias': False}
            no_bias_kwargs.update(kwargs)
            return compose(
                DarknetConv2D(*args, **no_bias_kwargs),
                BatchNormalization(),
                LeakyReLU(alpha=0.1))

        def compose(*funcs):
            """Compose arbitrarily many functions, evaluated left to right.

            Reference: https://mathieularose.com/function-composition-in-python/
            """
            # return lambda x: reduce(lambda v, f: f(v), funcs, x)
            if funcs:
                return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
            else:
                raise ValueError(
                    'Composition of empty sequence not supported.')

        x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3, 3)))(image_input)
        x2 = compose(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
        pred_yolo_1 = compose(
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            DarknetConv2D(pred_filter_count, (1, 1)))(x2)

        x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x2)

        pred_yolo_2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3, 3)),
            DarknetConv2D(pred_filter_count, (1, 1)))([x2, x1])

        self.pred_layers = [pred_yolo_1, pred_yolo_2]
        self.input_layers = [image_input]

        self.head_layers_cnt = 2
        self.downgrades = [32, 16]
