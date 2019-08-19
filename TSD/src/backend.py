from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Concatenate, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D, add, concatenate, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Keras_MobileNetV2
from tensorflow.keras.applications.xception import Xception as Keras_Xception

from functools import wraps, reduce


class DetBackend(object):
    def __init__(self):
        self.outputs = []
        self.input_layers = []

        self.head_layers_cnt = -1
        self.downgrades = [1]


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


class NewMobileNetV2(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        import mobilenet_utils as mnu

        alpha = 0.5

        channel_axis = 1 if mnu.K.image_data_format() == 'channels_first' else -1

        first_block_filters = mnu._make_divisible(32 * alpha, 8)

        x = mnu.Conv2D(first_block_filters, 3, padding='same',
                        strides=2, use_bias=False, name='Conv1')(image_input)
        x = mnu.BatchNormalization(axis=channel_axis, name='bn_Conv1')(x)
        x = mnu.ReLU(6., name='Conv1_relu')(x)

        x = mnu._inverted_residual_block(
            x, filters=16, kernel=3, t=1, strides=1, n=1, alpha=alpha, block_id=0)
        x = mnu._inverted_residual_block(
            x, filters=24, kernel=3, t=6, strides=2, n=2, alpha=alpha, block_id=1)
        x = mnu._inverted_residual_block(
            x, filters=32, kernel=3, t=6, strides=2, n=3, alpha=alpha, block_id=3)
        x = mnu._inverted_residual_block(
            x, filters=64, kernel=3, t=6, strides=2, n=4, alpha=alpha, block_id=6)
        x = mnu._inverted_residual_block(
            x, filters=96, kernel=3, t=6, strides=1, n=3, alpha=alpha, block_id=10)
        # x = mnu._inverted_residual_block(
        #     x, filters=160, kernel=3, t=6, strides=2, n=3, alpha=alpha, block_id=13)
        # x = mnu._inverted_residual_block(
        #     x, filters=320, kernel=3, t=6, strides=1, n=1, alpha=alpha, block_id=16)

        last_block_filters = mnu._make_divisible(1280 / 4 * alpha, 8)
        # last_block_filters = 1280
        x = mnu.Conv2D(last_block_filters, 1, padding='same',
                        strides=1, use_bias=False, name='Conv_1')(x)
        x = mnu.BatchNormalization(axis=channel_axis, name='Conv_1_bn')(x)
        x = mnu.ReLU(6., name='out_relu')(x)

        self.outputs = [x]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [16]


class MobileNetV2(DetBackend):
    def __init__(self, **kwargs):
        train_shape = kwargs['train_shape']

        image_input = Input(shape=train_shape, name='input_img')

        alpha = self.alpha

        mobilenetv2 = Keras_MobileNetV2(
            input_tensor=image_input, include_top=False, weights=None, alpha=alpha)
        x = mobilenetv2.output
    
        self.outputs = [x]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [32]


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
            no_bias_kwargs = {
                'use_bias': False
            }
            no_bias_kwargs.update(kwargs)
            return compose(DarknetConv2D(*args, **no_bias_kwargs),
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
            )(image_input)

        x2 = compose(
            DarknetConv2D_BN_Leaky(256, (3, 3)),        
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            )(x1)

        pred_yolo_1 = compose(
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            # DarknetConv2D_BN_Leaky(pred_filter_count, (1, 1), layer_idx=10, BN=False, relu=False, name='pred_1'),
            # DarknetConv2D(pred_filter_count, (1, 1), name='conv2d_10')
            )(x2)

        x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2)
            )(x2)

        pred_yolo_2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3, 3)),
            # DarknetConv2D_BN_Leaky(pred_filter_count, (1, 1), layer_idx=13, BN=False, relu=False, name='pred_2'),
            # DarknetConv2D(pred_filter_count, (1, 1), name='conv2d_13')
            )([x2, x1])

        self.outputs = [pred_yolo_1, pred_yolo_2]
        self.input_layers = [image_input]

        self.head_layers_cnt = 2
        self.downgrades = [32, 16]


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

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                   name='conv_1', use_bias=False)(image_input)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                   name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                   name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                   name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                   name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                   name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same',
                   name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                   name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                   name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same',
                   name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                   name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same',
                   name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                   name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                   name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same',
                   name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                   name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same',
                   name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                   name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                   name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                   name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(
            1, 1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                   name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Backend ends =)
        # pred_yolo_1 = Conv2D(pred_filter_count,
        #                      (1, 1),
        #                      strides=(1, 1),
        #                      padding='same',
        #                      name='DetectionLayer',
        #                      kernel_initializer='lecun_normal')(x)

        self.outputs = [x]
        self.input_layers = [image_input]

        self.head_layers_cnt = 1
        self.downgrades = [32]
