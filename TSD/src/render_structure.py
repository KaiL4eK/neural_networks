from utils.utils import preprocess_input, init_session
import cv2
import argparse
import tensorflow as tf
import keras.backend as K
import yolo 
import json
from tensorflow.keras.utils import plot_model
import os

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    init_session(0.5)

    train_model, infer_model, _ = yolo.create_model_new(
        nb_class            = 1,
        anchors_per_output  = config['model']['anchors_per_output'],
        max_input_size      = config['model']['max_input_size'],
        anchors             = config['model']['anchors'], 
        base                = config['model']['base'],
        train_shape         = (*config['model']['infer_shape'], 3),
        load_src_weights    = False
    )

    model_render_file = 'images/{}.png'.format(config['model']['base'])
    if not os.path.isdir(os.path.dirname(model_render_file)):
        os.makedirs(os.path.dirname(model_render_file))
    plot_model(infer_model, to_file=model_render_file, show_shapes=True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Help =)')
    argparser.add_argument('conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)

