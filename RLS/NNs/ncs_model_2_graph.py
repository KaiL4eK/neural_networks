import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from utils import makedirs
import os
import json

import shutil

from net import *

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

import numpy as np

NO_SIGMOID = True

G_SHAPE = (160, 320, 3)
input_img = np.ones(G_SHAPE, dtype='float32')

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)')
argparser.add_argument('-w', '--weights', help='weights path')

# Dont use concat
# Dont use sigmoid

def relu6(x):
    return K.relu(x, max_value=6)

def get_test_model(input_shape, lr):

    input = Input(input_shape, name='input_img')

    x = Conv2D(filters = 16, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(input)
    x = Activation(activation=relu6)(x)

    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(filters = 16, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    x = Activation(activation=relu6)(x)

    output_lin = Conv2D(1, 1)(x)
    output_sig = Activation(activation = 'sigmoid')(output_lin)

    train_model = Model(input, output_sig)
    infer_model = Model(input, output_lin)

    train_model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy', iou_metrics])

    return train_model, infer_model




def _main_(args):
    weights_path = args.weights
    output_fname = args.output

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    # train_model, infer_model = get_test_model(input_shape=G_SHAPE, lr=1e-4)
    train_model, infer_model = create_model(input_shape=G_SHAPE, lr=1e-4)

    if weights_path:
        train_model.load_weights(weights_path)

    model_inputs = [infer_model.input.name.split(':')[0]]
    model_outputs = [infer_model.output.name.split(':')[0]]

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs', 'laneseg')

        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        preds = train_model.predict(np.expand_dims(input_img, axis=0))
        host_output = preds[0]

        host_output_shape = host_output.shape
        host_flatten_shape = host_output.flatten().shape[0]

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

        # for n in frozen_graph.node:
            # print(n.name)

    print('Host Outputs: {} / {} -> {}'.format(model_outputs, host_output_shape, host_flatten_shape))

    frozen_graph_filename = output_fname
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    #####################################
    from subprocess import call

    process_args = ["mvNCCompile", output_fname, "-in", model_inputs[0], "-on", model_outputs[0], "-s", "12", "-o", "output/laneseg.graph"]
    call(process_args)

    process_args = ["mvNCProfile", output_fname, "-in", model_inputs[0], "-on", model_outputs[0], "-s", "12"]
    call(process_args)

    # for layer in infer_model.layers:
    #     print(layer.name)

    process_args = ["mvNCCheck", output_fname, "-in", model_inputs[0], "-on", model_outputs[0], "-s", "12"]
    call(process_args)

    # exit(1)

    #####################################

    import mvnc.mvncapi as fx

     # set the logging level for the NC API
    # fx.global_set_option(fx.GlobalOption.RW_LOG_LEVEL, 0)

    # get a list of names for all the devices plugged into the system
    devices = fx.enumerate_devices()
    if (len(devices) < 1):
        print("Error - no NCS devices detected, verify an NCS device is connected.")
        quit() 

    # get the first NCS device by its name.  For this program we will always open the first NCS device.
    dev = fx.Device(devices[0])
    
    # try to open the device.  this will throw an exception if someone else has it open already
    try:
        dev.open()
    except:
        print("Error - Could not open NCS device.")
        quit()


    print("Hello NCS! Device opened normally.")
    

    with open('output/laneseg.graph', mode='rb') as f:
        graphFileBuff = f.read()

    graph = fx.Graph('graph')

    print("FIFO Allocation")
    fifoIn, fifoOut = graph.allocate_with_fifos(dev, graphFileBuff)

    import time
    start = time.time()

    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, input_img, 'user object')
    ncs_output, userobj = fifoOut.read_elem()

    if NO_SIGMOID:
        ncs_output = sigmoid(ncs_output)
    # else:
    #     ncs_output = ncs_output[:host_flatten_shape]
    
    ncs_output = ncs_output.reshape(host_output_shape)

    print("Time: %.3f [ms]" % ((time.time() - start) * 1000))
    print('Comp1:', np.allclose(host_output, ncs_output, atol=0.003))
    print(host_output[0, 2])
    print(ncs_output[0, 2])
    print("NCS: ", ncs_output.shape, userobj)

    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()

    try:
        dev.close()
    except:
        print("Error - could not close NCS device.")
        quit()

    print("Goodbye NCS! Device closed normally.")
    print("NCS device working.")


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
