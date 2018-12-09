import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from utils import makedirs
import os
import json

import shutil

from net import iou_metrics, create_model

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

import numpy as np

G_SHAPE = (160, 320, 3)
input_img = np.ones(G_SHAPE, dtype='float32')

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)')
argparser.add_argument('-w', '--weights', help='weights path')

def get_test_model(input_shape, lr):
    inputs = Input(input_shape, name='input_img')



    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    output = Conv2D(1, 1, activation = 'sigmoid')(pool1)


    model = Model(inputs, output)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy', iou_metrics])

    return model



def _main_(args):
    weights_path = args.weights
    output_fname = args.output

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    infer_model = get_test_model(input_shape=G_SHAPE, lr=1e-4)

    output_shape = infer_model.output.shape

    model_outputs = [infer_model.output.name.split(':')[0]]
    

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs', 'laneseg')

        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        preds = infer_model.predict(np.expand_dims(input_img, axis=0))
        host_output = preds[0]
        host_output_shape = host_output.shape
        host_flatten_shape = host_output.flatten().shape[0]

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    print('Host Outputs: {} / {} -> {}'.format(model_outputs, host_output_shape, host_flatten_shape))

    frozen_graph_filename = output_fname
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    #####################################
    process_args = ["mvNCCompile", output_fname, "-ec", "-in", "input_img", "-on", model_outputs[0], "-s", "12", "-o", "output/laneseg.graph"]
    # print(process_args, '\n')

    from subprocess import call
    call(process_args)

    # process_args = ["mvNCProfile", output_fname, "-ec", "-in", "input_img", "-on", model_outputs[0], "-s", "12"]
    # call(process_args)

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
    output, userobj = fifoOut.read_elem()

    ncs_output = output[:host_flatten_shape]
    ncs_output = ncs_output.reshape(host_output_shape)

    print('Comp1:', np.allclose(host_output, ncs_output, atol=0.001))

    print(host_output, ncs_output)

    print("Time: %.3f [ms]" % ((time.time() - start) * 1000))
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
