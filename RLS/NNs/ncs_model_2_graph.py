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
import config

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-w', '--weights', help='weights path')

def _main_(args):
    weights_path = args.weights

    output_pb_fpath = config.NCS_GRAPH

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    network_input_shp = (config.NETWORK_INPUT_H, config.NETWORK_INPUT_W, config.NETWORK_INPUT_C)
    train_model, infer_model = create_model(input_shape=network_input_shp, lr=1e-4)

    if weights_path:
        train_model.load_weights(weights_path)

    model_inputs = [infer_model.input.name.split(':')[0]]
    model_outputs = [infer_model.output.name.split(':')[0]]

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs', 'laneseg_graph')
        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)
        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    with open(output_pb_fpath, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    #####################################
    from subprocess import call

    process_args = ["mvNCCompile", output_pb_fpath, "-in", model_inputs[0], "-on", model_outputs[0], "-s", "12", "-o", config.NCS_GRAPH]
    call(process_args)

    process_args = ["mvNCProfile", output_pb_fpath, "-in", model_inputs[0], "-on", model_outputs[0], "-s", "12"]
    call(process_args)

    process_args = ["mvNCCheck", output_pb_fpath, "-in", model_inputs[0], "-on", model_outputs[0], "-s", "12"]
    # call(process_args)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
 
