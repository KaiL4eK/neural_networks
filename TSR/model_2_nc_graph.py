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
    weights_path = args.weights
    output_fname = args.output
    config_path = args.conf

    output_dpath = os.path.dirname(output_fname)
    if not os.path.isdir(output_dpath):
        os.makedirs(output_dpath)

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    net_input_shape = (config['model']['input_side_sz'],
                       config['model']['input_side_sz'],
                       3)

    classes = data.get_classes(config['train']['cache_name'])

    if not classes:
        print('Failed to get classes list')

    mvnc_model = models.create(
        base            = config['model']['base'],
        num_classes     = len(classes),
        input_shape     = net_input_shape)

    print_summary(mvnc_model)

    if weights_path:
        mvnc_model.load_weights(weights_path)

    model_output_names = [mvnc_model.output.name.split(':')[0]]
    model_outputs = [mvnc_model.output]

    print('Outputs: {}'.format(model_outputs))
    # print('Output shapes: {}'.format(model_output_shapes))

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        for op in graphdef.node:
            print(op.name)

        dirpath = os.path.join('logs', config['model']['base'])

        shutil.rmtree(dirpath, ignore_errors=True)

        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_output_names)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_fname
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    K.clear_session()

    # mvNCCheck nc/mobilenetv2.pb -in input_img -on out_reshape/Reshape -s 12
    # mvNCProfile nc/mobilenetv2.pb -in input_img -on out_reshape/Reshape -s 12

if __name__ == '__main__':
    _main_()
