import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_graph_pb", required=True)
    parser.add_argument("-o", "--output_events_path", required=True)
    args = vars(parser.parse_args())

    with tf.Session() as sess:
        model_filename = args['input_graph_pb']

        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)

        logdir=args['output_events_path']
        train_writer = tf.summary.FileWriter(logdir)
        train_writer.add_graph(sess.graph)

        print([n.name for n in sess.graph.as_graph_def().node])

    print("Model Imported. Visualize by running > tensorboard --logdir={}".format(logdir))