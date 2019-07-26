from utils.utils import preprocess_input, init_session
import cv2
import argparse
import tensorflow as tf
import keras.backend as K
import yolo 
import json
from keras.utils import plot_model


def _main_(args):
    config_path = args.conf
    input_img   = args.input

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    init_session(0.5)

    train_model, infer_model, freeze_num = yolo.create_model(
        nb_class            = 1,
        anchors             = config['model']['anchors'], 
        base                = config['model']['base'],
        train_shape         = (416, 416, 3),
        load_src_weights    = False
    )

    plot_model(train_model, to_file='model_{}.png'.format(config['model']['base']), show_shapes=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)

