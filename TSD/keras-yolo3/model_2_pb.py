import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import tensorrt.parsers.uffparser as uffparser
import uff

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output frozen graph (.pb file)')
argparser.add_argument('-w', '--weights', help='weights path')


def _main_(args):
    weights_path = args.weights
    output_fname = args.output

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    infer_model = load_model( weights_path )

    model_output1 = 'conv2d_10/BiasAdd'
    model_output2 = 'conv2d_13/BiasAdd'


    model_output1 = 'DetectionLayer1/BiasAdd'
    model_output2 = 'DetectionLayer2/BiasAdd'
    model_outputs = [model_output1, model_output2]

    # model_output1 = 'conv_81/BiasAdd'
    # model_output2 = 'conv_93/BiasAdd'
    # model_output3 = 'conv_105/BiasAdd'

    # model_outputs = [model_output1, model_output2, model_output3]

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        # for op in sess.graph.get_operations():
            # print(op.name)

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_fname
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    uff_model = uff.from_tensorflow(frozen_graph, model_outputs)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
