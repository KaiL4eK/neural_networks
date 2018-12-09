import os

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def init_gpu_session(rate):

    import keras.backend as K
    import tensorflow as tf

    config = tf.ConfigProto()

    config.gpu_options.allow_growth                     = True
    config.gpu_options.per_process_gpu_memory_fraction  = rate

    K.set_session(tf.Session(config=config))

def print_pretty(str):
	print('-'*30)
	print(str)
	print('-'*30)
