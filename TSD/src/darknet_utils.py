from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.regularizers import l2

from functools import wraps, reduce

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    conv_kwargs = {}
    conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **conv_kwargs)

# He_uniform better use with ReLU (let`s check this!)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {
        'use_bias': False,
        'kernel_initializer': 'he_uniform'
    }
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))


def Darknet19Conv2D_BN_Leaky(*args, idx, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    new_kwargs = {
        'use_bias': False,
        'kernel_initializer': 'he_uniform',
        'name': 'conv_' + str(idx)
    }
    new_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **new_kwargs),
                   BatchNormalization(name='norm_'+str(idx)),
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


def TinyV3Base(x):
    x8 = compose(
        DarknetConv2D_BN_Leaky(16, 3),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        # x2
        DarknetConv2D_BN_Leaky(32, 3),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        # x4
        DarknetConv2D_BN_Leaky(64, 3),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        # x8
    )(x)

    x16 = compose(
        DarknetConv2D_BN_Leaky(128, 3),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        # x16
    )(x8)

    x32 = compose(
        DarknetConv2D_BN_Leaky(256, 3),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        # x32
        DarknetConv2D_BN_Leaky(512, 3),
        MaxPooling2D(pool_size=2, strides=1, padding='same'),
        DarknetConv2D_BN_Leaky(1024, 3),
        DarknetConv2D_BN_Leaky(256, 1),
    )(x16)

    return x8, x16, x32
