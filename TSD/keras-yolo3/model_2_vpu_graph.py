import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import tensorrt.parsers.uffparser as uffparser
import uff
from utils.utils import makedirs
import os
import json
from yolo import create_model

import shutil

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-c', '--conf', help='path to configuration file')  


def _main_(args):
    weights_path = args.weights
    output_fname = args.output
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)


    _, infer_model, mvnc_model, _ = create_model(
        nb_class            = 1,
        anchors             = config['model']['anchors'],
        base                = config['model']['base'],
        load_src_weights    = False,
        train_shape         = (416, 416, 3)
    )

    mvnc_model.load_weights(weights_path)

    model_outputs = [mvnc_model.output.name.split(':')[0]]

    print('Outputs: {}'.format(model_outputs))

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs', config['model']['base'])

        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_fname
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
