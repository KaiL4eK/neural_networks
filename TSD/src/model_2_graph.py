import tensorflow as tf
import tensorflow.keras.backend as K
from utils.utils import makedirs
import os
import json
import yolo
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import shutil
from _common import utils

try:
    import uff
    uff_imported = True
except:
    uff_imported = False
    print('Failed to import UFF')

import argparse
argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-n', '--ncs', action='store_true', help='enable NCS files generation')
args = argparser.parse_args()


def _main_():
    weights_path = args.weights
    config_path = args.conf
    ncs_flag = args.ncs

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    output_pb_fpath = utils.get_pb_graph_fpath(config)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    train_sz = config['model']['infer_shape']

    labels = ['sign']

    config['model']['labels'] = labels
    yolo_model = yolo.YOLO_Model(
        config['model']
    )

    if weights_path:
        yolo_model.load_weights(weights_path)

    infer_model = yolo_model.infer_model

        # infer_model = load_model(weights_path)
        # image_input = Input(shape=(train_sz[0], train_sz[1], 3), name='input_img')
        # infer_model = Model(image_input, infer_model(image_input))
    # else:

    model_input_names = [infer_model.input.name.split(':')[0]]
    model_output_names = [out.name.split(':')[0] for out in infer_model.output]

    print('Model:')
    print('  Inputs: {}'.format(model_input_names))
    print('  Outputs: {}'.format(model_output_names))

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs/tf_export', config['model']['base'])

        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_output_names)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_pb_fpath
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    K.clear_session()
    print('Frozen graph done!')

    if uff_imported:
        uff_model = uff.from_tensorflow(frozen_graph, model_output_names)

    if ncs_flag:
        from subprocess import call

        if weights_path:
            graph_fpath = utils.get_ncs_graph_fpath(config)
            print('    Writing to {}'.format(graph_fpath))

            process_args = ["mvNCCompile", output_pb_fpath, "-in", model_input_names[0], "-on", model_output_names[0], "-s", "12",
                            "-o", graph_fpath]
            call(process_args)

        # print('    Compiled, check performance')

        process_args = ["mvNCProfile", output_pb_fpath, "-in", model_input_names[0], "-on", model_output_names[0], "-s", "12"]
        call(process_args)

        process_args = ["mvNCCheck", output_pb_fpath, "-in", model_input_names[0], "-on", model_output_names[0], "-s", "12"]
        call(process_args)


if __name__ == '__main__':
    _main_()
