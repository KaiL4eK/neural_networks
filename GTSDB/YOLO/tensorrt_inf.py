import tensorrt
import argparse
import json
import cv2
import numpy as np

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

    model_output = 'YOLO_output/Reshape'
    model_output = 'DetectionLayer/BiasAdd'

    try:
        yolo2_engine = tensorrt.lite.Engine(framework="tf",
                                            path=pb_path,
                                            max_batch_size=1,
                                            input_nodes={'input_1': (3, config['model']['input_size_w'], config['model']['input_size_h'])},
                                            output_nodes=[model_output])  # Output layers
    except:
        print('Failed to create engine')
        exit(1)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (config['model']['input_size_w'], config['model']['input_size_h']))
    image = np.array(image).reshape((3, config['model']['input_size_w'], config['model']['input_size_h']))
    image = image.astype('float32')

    image = image / 255.
    image = image - 0.5
    image = image * 2.

    result = yolo2_engine.infer([image])

    print(np.array(result).shape)

    result = np.array(result).reshape((13, 13, 5, 6))

    print(result.shape)

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
