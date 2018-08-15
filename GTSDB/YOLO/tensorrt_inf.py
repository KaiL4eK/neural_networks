import uff
import tensorrt
import argparse
import json

argparser = argparse.ArgumentParser(
    description='TensorRT based inference for network')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to PB (frozen TF graph) converted file')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def _main_(args):

    config_path  = args.conf
    pb_path      = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    yolo2_engine = tensorrt.lite.Engine(framework="tf",
                                        path=pb_path,
                                        max_batch_size=10,
                                        input_nodes={'input_1': (3, 416, 416)},
                                        output_nodes=['YOLO_output/Reshape'])  # Output layers

    #
    # stream = uff.from_tensorflow_frozen_model(pb_path, ['YOLO_output/Reshape'])
    #
    # mnist_engine = tensorrt.lite.Engine(framework='uff',
    #                                     stream=stream,
    #                                     input_nodes={'input_1': (config['model']['input_size_h'],
    #                                                         config['model']['input_size_w'], 3)},
    #                                     output_nodes=['YOLO_output/Reshape'])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
