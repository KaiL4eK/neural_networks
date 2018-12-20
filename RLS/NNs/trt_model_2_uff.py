import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.utils.layer_utils import print_summary
import tensorrt.parsers.uffparser as uffparser
import uff
from utils import makedirs
import os
import json

from net import *
import shutil
import config

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)', default='laneseg')
argparser.add_argument('-w', '--weights', help='weights path')

def _main_(args):
    weights_path = args.weights
    output_fname = args.output

    uff_fname = output_fname + '.uff'

    # output_frozen_pb_fpath = os.path.join('frozen_pb', frozen_graph_fname)
    output_uff_fpath = os.path.join('TensorRT/uff', uff_fname)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    network_input_shp = (config.NETWORK_INPUT_H, config.NETWORK_INPUT_W, config.NETWORK_INPUT_C)
    train_model, infer_model = create_model(input_shape=network_input_shp, lr=1e-4)

    print_summary(train_model)

    if weights_path:
        train_model.load_weights(weights_path)

    model_inputs = [train_model.input.name.split(':')[0]]
    model_outputs = [train_model.output.name.split(':')[0]]

    print(model_inputs)
    print(model_outputs)

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs', 'laneseg_graph')
        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)
        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    # frozen_graph_filename = output_frozen_pb_fpath
    # with open(frozen_graph_filename, 'wb') as f:
    #     f.write(frozen_graph.SerializeToString())
    # f.close()

    uff_model = uff.from_tensorflow(frozen_graph, model_outputs, output_filename=output_uff_fpath)



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
