from utils.utils import preprocess_input, init_session
import cv2
import argparse
import tensorflow as tf
import keras.backend as K
from yolo import create_model
import json
from keras.utils import plot_model


def _main_(args):
    config_path = args.conf
    input_img   = args.input

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    init_session(0.5)

    if input_img:    
        img = cv2.imread(input_img)

        img = preprocess_input(img, 416, 416)

        cv2.imshow('1', img)
        cv2.waitKey(0)


    train_model, infer_model, freeze_num = create_model(
        nb_class            = 1, 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = 1, 
        max_input_size      = config['model']['max_input_size'], 
        batch_size          = config['train']['batch_size'], 
        warmup_batches      = 0,
        ignore_thresh       = config['train']['ignore_thresh'],
        multi_gpu           = 1,
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
        base                = config['model']['base'],
        img_shape           = (416, 416, 3),
        load_src_weights    = False
    )


    # train_model.summary()
    # infer_model.summary()

    # infer_model.layers[249].trainable = False
    # infer_model.layers[250].trainable = False
    # infer_model.layers[251].trainable = False

    # for i, layer in enumerate(train_model.layers):
        # print('{} / {} / {}'.format(i, layer.name, layer.trainable))

    plot_model(train_model, to_file='model_{}.png'.format(config['model']['base']), show_shapes=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   
    argparser.add_argument('-i', '--input', help='path to input file', default=None)   

    args = argparser.parse_args()
    _main_(args)

