import tensorflow as tf
import keras.backend as K
from utils.utils import makedirs
import os
import json
from yolo import create_model
from keras.utils.layer_utils import print_summary
from keras.models import load_model
import shutil
from keras.layers import Input
from keras.models import Model

import argparse
argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-t', '--test', action='store_true', help='path to configuration file')
args = argparser.parse_args()


def _main_():
    test_perf = args.test
    weights_path = args.weights
    output_pb_fpath = args.output
    config_path = args.conf

    

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    train_sz = config['model']['infer_shape']

    if weights_path:
        mvnc_model = load_model(weights_path)
        image_input = Input(shape=(train_sz[0], train_sz[1], 3), name='input_img')
        mvnc_model_output = mvnc_model(image_input)

        mvnc_model = Model(image_input, mvnc_model_output)
    else:
        _, _, mvnc_model, _ = create_model(
            nb_class=6,
            anchors=config['model']['anchors'],
            base=config['model']['base'],
            load_src_weights=False,
            train_shape=(train_sz[0], train_sz[1], 3)
        )

    print_summary(mvnc_model)
    from keras.utils.vis_utils import plot_model
    model_render_file = 'images/{}.png'.format(config['model']['base'])
    if not os.path.isdir(os.path.dirname(model_render_file)):
        os.makedirs(os.path.dirname(model_render_file))
    plot_model(mvnc_model, to_file=model_render_file, show_shapes=True)

    # mvnc_model.load_weights(weights_path)

    model_input_names = [mvnc_model.input.name.split(':')[0]]
    model_output_names = [mvnc_model.output.name.split(':')[0]]

    print('Outputs: {}'.format(model_output_names))

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs', config['model']['base'])

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

    #####################################
    from subprocess import call

    graph_fpath = output_pb_fpath + '.graph'
    print('    Writing to {}'.format(graph_fpath))

    if weights_path:
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
