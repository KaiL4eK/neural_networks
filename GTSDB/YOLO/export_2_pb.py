import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

import argparse
import os
from frontend import YOLO
import json


argparser = argparse.ArgumentParser(
    description='Convert Keras model to UFF format')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')


def _main_(args):
    config_path  = args.conf
    weights_path = args.weights

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    if weights_path == '':
        weights_path = config['train']['saved_weights_name']

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'],config['model']['input_size_w']), 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                gray_mode           = config['model']['gray_mode'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    model = yolo.model

    # model_input1 = model.input[0].name.strip(':0')
    # model_input2 = model.input[1].name.strip(':0')
    # model_output = model.output.name.strip(':0')
    # print(model_input1, model_input2, model_output)
    #
    # print('Now TensorRT time!')
    #
    # graph = tf.get_default_graph().as_graph_def()
    #
    # print('Got graph')
    #
    # # Get session
    # sess = tf.keras.backend.get_session()
    #
    # print('Got session')
    #
    # sess.run(tf.global_variables_initializer())
    #
    # print('Got initialized')
    #
    # # freeze graph and remove nodes used for training
    # frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, [model_output])
    #
    # print('Constants done')
    #
    # frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    #
    # print('Got frozen graph')
    #
    # # Create UFF model and dump it on disk
    # uff_model = uff.from_tensorflow(frozen_graph, [model_output])
    # dump = open('output/tiny_yolo.uff', 'wb')
    # dump.write(uff_model)
    # dump.close()

    K.set_learning_phase(0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
        checkpoint_path = saver.save(sess, 'output/saved_ckpt', global_step=0, latest_filename='checkpoint_state')
        graph_io.write_graph(sess.graph, 'output', 'tmp.pb')

        print('Graph saved')

        freeze_graph.freeze_graph('output/tmp.pb', '',
                                  False, checkpoint_path, model.output.name.strip(':0'),
                                  "save/restore_all", "save/Const:0",
                                  'output/model.pb', False, "")

        os.unlink('output/tmp.pb')


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)