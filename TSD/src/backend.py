from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Concatenate, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D, add, concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Keras_MobileNetV2
from tensorflow.keras.applications.xception import Xception as Keras_Xception

import mobilenet_utils as mnu
import darknet_utils as dnu

class DetBackend(object):
    def __init__(self):
        pass
        # self.outputs = []
        # self.input_layers = []

        # self.head_layers_cnt = -1
        # self.downgrades = [1]


# 6,6, 7,7, 8,8, 10,10, 12,11, 14,14, 18,17, 23,22, 32,31
# 6,6, 9,9, 12,12, 15,15, 21,20, 31,30
# 8,8, 14,13, 25,24
class NewMobileNetV2(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']
        base_params = kwargs['base_params']

        image_input = Input(shape=train_shape, name='input_img')

        alpha = base_params['alpha']

        channel_axis = 1 if mnu.K.image_data_format() == 'channels_first' else -1

        first_block_filters = mnu._make_divisible(32 * alpha, 8)
        x = mnu.Conv2D(first_block_filters, 3, padding='same', strides=2, use_bias=False, name='Conv1')(image_input)
        x = mnu.BatchNormalization(axis=channel_axis, name='bn_Conv1')(x)
        x = mnu.ReLU(6., name='Conv1_relu')(x)

        x = mnu._inverted_residual_block(x, 16, 3, t=1, s=1, n=1, alpha=alpha, block_id=0)
        x = mnu._inverted_residual_block(x, 24, 3, t=6, s=2, n=2, alpha=alpha, block_id=1)
        x4 = x
        x = mnu._inverted_residual_block(x, 32, 3, t=6, s=2, n=3, alpha=alpha, block_id=3)
        x8 = x
        x = mnu._inverted_residual_block(x, 64, 3, t=6, s=2, n=4, alpha=alpha, block_id=6)
        x = mnu._inverted_residual_block(x, 96, 3, t=6, s=1, n=3, alpha=alpha, block_id=10)
        x16 = x

        # x = mnu._inverted_residual_block(x, filters=160, kernel=3, t=6, strides=2, n=3, alpha=alpha, block_id=13)
        # x = mnu._inverted_residual_block(x, filters=320, kernel=3, t=6, strides=1, n=1, alpha=alpha, block_id=16)

        last_block_filters = mnu._make_divisible(1280 * alpha, 8)
        x = mnu.Conv2D(last_block_filters, 1, padding='same', strides=1, use_bias=False, name='out_conv1')(x16)
        x = mnu.BatchNormalization(axis=channel_axis, name='out_bn1')(x)
        y1 = mnu.ReLU(6., name='out_relu1')(x)

        # Next branch
        x8u = UpSampling2D(2)(x16)
        x = Concatenate()([x8u, x8])

        x = mnu._inverted_residual_block(x, 32, 3, t=6, s=1, n=2, alpha=alpha, block_id=13)

        last_block_filters = mnu._make_divisible(1280 / 2 * alpha, 8)
        x = mnu.Conv2D(last_block_filters, 1, padding='same', strides=1, use_bias=False, name='out_conv2')(x)
        x = mnu.BatchNormalization(axis=channel_axis, name='out_bn2')(x)
        y2 = mnu.ReLU(6., name='out_relu2')(x)

        self.outputs = [y1, y2]
        self.inputs = [image_input]

        self.head_layers_cnt = 2
        self.downgrades = [16, 8]

        self.model = Model(*self.inputs, *self.outputs)


class MadNetv1(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')
        x = image_input

        x2 = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(4, 5, strides=(4, 4)),
            # MaxPooling2D(pool_size=2, strides=2, padding='same')
            )(x)

        x16 = dnu.compose(
            # dnu.DarknetConv2D_BN_Leaky(16, 3),
            # MaxPooling2D(pool_size=2, strides=2, padding='same'),
            dnu.DarknetConv2D_BN_Leaky(32, 3),
            MaxPooling2D(pool_size=2, strides=2, padding='same'),
            dnu.DarknetConv2D_BN_Leaky(64, 3),
            MaxPooling2D(pool_size=2, strides=2, padding='same'),
        )(x2)

        x32 = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(128, 3, dilation_rate=1),
            dnu.DarknetConv2D_BN_Leaky(128, 3, dilation_rate=2),
            # dnu.DarknetConv2D_BN_Leaky(128, 3, dilation_rate=2),
            MaxPooling2D(pool_size=2, strides=2, padding='same'),
        )(x16)

        x32 = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(256, 3, dilation_rate=2),
            dnu.DarknetConv2D_BN_Leaky(256, 3, dilation_rate=4),
            # dnu.DarknetConv2D_BN_Leaky(256, 3, dilation_rate=4),

            # DarknetConv2D_BN_Leaky(512, 3),
            # MaxPooling2D(pool_size=2, strides=1, padding='same'),
            # DarknetConv2D_BN_Leaky(1024, 3),
            # DarknetConv2D_BN_Leaky(256, 1),
        )(x32)


        # y1 = dnu.compose(
        #     dnu.DarknetConv2D_BN_Leaky(512, 3),
        #     )(x64)

        # x32u = dnu.compose(
        #     dnu.DarknetConv2D_BN_Leaky(128, 1),
        #     UpSampling2D(2)
        #     )(x64)

        y2 = dnu.compose(
            # Concatenate(),
            dnu.DarknetConv2D_BN_Leaky(256, 3),
            # )([x32u, x32])
            )(x32)

        x16u = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(128, 1),
            UpSampling2D(2)
            # )(y2)
            )(x32)

        y3 = dnu.compose(
            Concatenate(),
            dnu.DarknetConv2D_BN_Leaky(128, 3),
            )([x16u, x16])

        self.outputs = [y2, y3]
        self.inputs = [image_input]

        self.head_layers_cnt = 2
        self.downgrades = [32, 16]

class SqueezeNet(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            # x     = LeakyReLU(name=s_id + relu + sq1x1)(x)
            x = ReLU(6., name=s_id + relu + sq1x1)(x)

            left = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            # left  = LeakyReLU(name=s_id + relu + exp1x1)(left)
            left = ReLU(6., name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            # right = LeakyReLU(name=s_id + relu + exp3x3)(right)
            right = ReLU(6., name=s_id + relu + exp3x3)(right)

            x = add([left, right], name=s_id + 'concat')

            return x

        x = Conv2D(64, (3, 3), strides=(2, 2),
                padding='same', name='conv1')(image_input)
        # x = LeakyReLU(name='relu_conv1')(x)
        x = ReLU(6., name='relu_conv1')(x)

        x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)

        x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)

        x = MaxPooling2D(pool_size=(2, 2), name='pool5')(x)

        # x = ZeroPadding2D(padding=(1, 1))(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)

        x = MaxPooling2D(pool_size=(2, 2), name='pool7')(x)

        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.outputs = [x]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [32]


class Xception(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        xception = Keras_Xception(input_tensor=image_input,
                                  include_top=False,
                                  weights='imagenet')

        self.outputs = [xception.output]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [32]


class MobileNetV2(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']
        base_params = kwargs['base_params']

        image_input = Input(shape=train_shape, name='input_img')

        alpha = base_params['alpha']

        mobilenetv2 = Keras_MobileNetV2(input_tensor=image_input, include_top=False, weights=None, alpha=alpha)
        x = mobilenetv2.output
    
        self.outputs = [x]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [32]


class Tiny_YOLOv3(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        x8, x16, x32 = dnu.TinyV3Base(image_input)

        y1 = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(512, 3),
            )(x32)

        x16u = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(128, 1),
            UpSampling2D(2)
            )(x32)

        y2 = dnu.compose(
            Concatenate(),
            dnu.DarknetConv2D_BN_Leaky(256, 3),
            )([x16u, x16])

        self.outputs = [y1, y2]
        self.input_layers = [image_input]

        self.head_layers_cnt = 2
        self.downgrades = [32, 16]


class Small_Tiny_YOLOv3(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        x8, x16, x32 = dnu.TinyV3Base(image_input)

        y1 = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(512, 3),
            )(x32)

        x16u = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(128, 1),
            UpSampling2D(2)
            )(x32)

        y2 = dnu.compose(
            Concatenate(),
            dnu.DarknetConv2D_BN_Leaky(256, 3),
            )([x16u, x16])

        x8u = dnu.compose(
            dnu.DarknetConv2D_BN_Leaky(128, 1),
            UpSampling2D(2)
            # )(y2)
            )(x16u)

        y3 = dnu.compose(
            Concatenate(),
            dnu.DarknetConv2D_BN_Leaky(128, 3),
            )([x8u, x8])

        self.outputs = [y1, y2, y3]
        self.inputs = [image_input]

        self.head_layers_cnt = 3
        self.downgrades = [32, 16, 8]


class Darknet53(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        def _conv_block(inp, convs, do_skip=True):
            x = inp
            count = 0

            for conv in convs:
                if count == (len(convs) - 2) and do_skip:
                    skip_connection = x
                count += 1

                if conv['stride'] > 1:
                    # unlike tensorflow darknet prefer left and top paddings
                    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
                x = Conv2D(conv['filter'],
                           conv['kernel'],
                           strides=conv['stride'],
                           # unlike tensorflow darknet prefer left and top paddings
                           padding='valid' if conv['stride'] > 1 else 'same',
                           name='conv_' + str(conv['layer_idx']),
                           use_bias=False if conv['bnorm'] else True)(x)
                if conv['bnorm']:
                    x = BatchNormalization(
                        epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
                if conv['leaky']:
                    x = LeakyReLU(alpha=0.1, name='leaky_' +
                                  str(conv['layer_idx']))(x)

            return add([skip_connection, x]) if do_skip else x

        # Layer  0 => 4
        x = _conv_block(image_input, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                      {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                      {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                      {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

        # Layer  5 => 8
        x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                            {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

        # Layer  9 => 11
        x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

        # Layer 12 => 15
        x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

        # Layer 16 => 36
        for i in range(7):
            x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])

        skip_36 = x

        # Layer 37 => 40
        x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

        # Layer 41 => 61
        for i in range(7):
            x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])

        skip_61 = x

        # Layer 62 => 65
        x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

        # Layer 66 => 74
        for i in range(3):
            x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])

        # Layer 75 => 79
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], do_skip=False)

        # Layer 80 => 82
        pred_yolo_1 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                                    #   {'filter': pred_filter_count, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}
                                      ], do_skip=False)

        # Layer 83 => 86
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], do_skip=False)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_61])

        # Layer 87 => 91
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], do_skip=False)

        # Layer 92 => 94
        pred_yolo_2 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                                    #   {'filter': pred_filter_count, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}
                                      ], do_skip=False)

        # Layer 95 => 98
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], do_skip=False)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_36])

        # Layer 99 => 106
        pred_yolo_3 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                                      {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                                      {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                                      {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                                      {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                                      {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                                    #   {'filter': pred_filter_count, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}
                                      ], do_skip=False)

        self.outputs = [pred_yolo_1, pred_yolo_2, pred_yolo_3]
        self.input_layers = [image_input]

        self.head_layers_cnt = 3
        self.downgrades = [32, 16, 8]


class Darknet19(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        def space_to_depth_x2(x):
            import tensorflow as tf
            return tf.space_to_depth(x, block_size=2)

        x = dnu.Darknet19Conv2D_BN_Leaky(32, 3, idx=1)(image_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = dnu.Darknet19Conv2D_BN_Leaky(64, 3, idx=2)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = dnu.Darknet19Conv2D_BN_Leaky(128, 3, idx=3)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(64, 1, idx=4)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(128, 3, idx=5)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = dnu.Darknet19Conv2D_BN_Leaky(256, 3, idx=6)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(128, 1, idx=7)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(256, 3, idx=8)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = dnu.Darknet19Conv2D_BN_Leaky(512, 3, idx=9)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(256, 1, idx=10)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(512, 3, idx=11)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(256, 1, idx=12)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(512, 3, idx=13)(x)
        skip_connection = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = dnu.Darknet19Conv2D_BN_Leaky(1024, 3, idx=14)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(512, 1, idx=15)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(1024, 3, idx=16)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(512, 1, idx=17)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(1024, 3, idx=18)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(1024, 3, idx=19)(x)
        x = dnu.Darknet19Conv2D_BN_Leaky(1024, 3, idx=20)(x)

        skip_connection = dnu.Darknet19Conv2D_BN_Leaky(64, 1, idx=21)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        x = dnu.Darknet19Conv2D_BN_Leaky(1024, 3, idx=22)(x)
        
        self.outputs = [x]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [32]
