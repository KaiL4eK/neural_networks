import argparse
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import tensorrt.parsers.uffparser as uffparser
import uff

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-o', '--output', help='path output graph')
argparser.add_argument('-w', '--weights', help='weights path')


def _main_(args):
    weights_path = args.weights
    output_fname = args.output

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    infer_model = load_model( weights_path )

    with K.get_session() as sess:

        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, "vpu_output/" + output_fname)

        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.close()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)


