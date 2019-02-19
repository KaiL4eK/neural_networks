import argparse
import tensorflow as tf
import keras.backend as K
import models
import data
from keras.models import load_model
import os
import json
from keras.utils.layer_utils import print_summary

import shutil

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-c', '--conf', help='path to configuration file')
args = argparser.parse_args()


def _main_():
    config_path = args.conf
    weights_path = args.weights
    output_pb_fpath = args.output
    config_path = args.conf

    output_dpath = os.path.dirname(output_pb_fpath)
    if not os.path.isdir(output_dpath):
        os.makedirs(output_dpath)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    K.set_learning_phase(0)

    # Swapped as net input -> [H x W x C]
    net_input_shape = (config['model']['input_shape'][1],
                       config['model']['input_shape'][0],
                       3)

    mvnc_model = models.create(
        base=config['model']['base'],
        input_shape=net_input_shape)

    if weights_path:
        mvnc_model.load_weights(weights_path)

    print_summary(mvnc_model)

    model_input_names = [mvnc_model.input.name.split(':')[0]]

    model_output_names = [mvnc_model.output.name.split(':')[0]]
    model_outputs = [mvnc_model.output]

    print('Outputs: {}'.format(model_outputs))
    # print('Output shapes: {}'.format(model_output_shapes))

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        # for op in graphdef.node:
            # print(op.name)

        dirpath = os.path.join('logs', config['model']['base'])

        shutil.rmtree(dirpath, ignore_errors=True)

        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_output_names)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_pb_fpath
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    K.clear_session()

    #####################################
    from subprocess import call

    graph_fpath = output_pb_fpath + '.graph'
    print('    Writing to {}'.format(graph_fpath))

    process_args = ["mvNCCompile", output_pb_fpath, "-in", model_input_names[0], "-on", model_output_names[0], "-s", "12",
                    "-o", graph_fpath]
    # call(process_args)

    print('    Compiled, check performance')

    process_args = ["mvNCCheck", output_pb_fpath, "-in", model_input_names[0], "-on", model_output_names[0], "-s", "12"]
    call(process_args)

    process_args = ["mvNCProfile", output_pb_fpath, "-in", model_input_names[0], "-on", model_output_names[0], "-s", "12"]
    call(process_args)

if __name__ == '__main__':
    _main_()
