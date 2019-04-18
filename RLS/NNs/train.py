from __future__ import print_function

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils.layer_utils import print_summary
from keras.utils import plot_model

from data import robofest_data_get_samples_preprocessed
from net import create_model
import argparse

from utils import print_pretty, init_gpu_session
import config

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-s', '--structure', action='store_true', help='Show only structure of network')

args = parser.parse_args()

init_gpu_session(1.0)


def train_and_predict():
    
    pretrained_weights_path = args.weights
    structure_mode = args.structure

    print_pretty('Creating and compiling model...')

    network_input_shp = (config.NETWORK_INPUT_H, config.NETWORK_INPUT_W, config.NETWORK_INPUT_C)
    output_shp = (config.NETWORK_INPUT_H, config.NETWORK_INPUT_W, 1)

    train_model, infer_model = create_model(input_shape=network_input_shp, lr=1e-4)

    if structure_mode:
        print_summary(train_model)
        plot_model(infer_model, rankdir='LB', show_shapes=True, show_layer_names=True, to_file='train_model.png')
        return

    if pretrained_weights_path:
        train_model.load_weights(args.weights)

    print_pretty('Loading and preprocessing train data...')

    train_imgs, train_masks, valid_imgs, valid_masks = robofest_data_get_samples_preprocessed(network_input_shp, output_shp)

    if len( np.unique(valid_masks) ) > 2:
        print('Valid: Preprocessing created mask with more than two binary values')
        exit(1)

    if len( np.unique(train_masks) ) > 2:
        print('Train: Preprocessing created mask with more than two binary values')
        exit(1)

    print_pretty('Setup data generator...')

    imgs_valid      = valid_imgs
    imgs_mask_valid = valid_masks

    imgs_train      = train_imgs
    imgs_mask_train = train_masks

    print('Train:', imgs_train.shape)
    print('Valid:', imgs_valid.shape)

    data_gen_args = dict( rotation_range=5,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          zoom_range=0.1,
                          horizontal_flip=True,
                          fill_mode='constant',
                          cval=0 )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    batch_size = 8
    image_datagen.fit(imgs_train, augment=True, seed=seed)
    mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)

    print_pretty('Flowing data...')

    image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, seed=seed)
    mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, seed=seed)

    print_pretty('Zipping generators...')

    train_generator = zip(image_generator, mask_generator)

    print_pretty('Fitting model...')

    checkpoint_vloss = CustomModelCheckpoint(
        model_to_save   = infer_model,
        filepath        = config.NET_BASENAME+'_ep{epoch:03d}-iou{iou_metrics:.3f}-val_iou{val_iou_metrics:.3f}'+'.h5',
        monitor         = 'val_loss', 
        verbose         = 1,
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )

    train_model.fit_generator( train_generator, steps_per_epoch=20, epochs=10000, verbose=1, validation_data=(imgs_valid, imgs_mask_valid),
        callbacks=[ModelCheckpoint('chk/weights_best.h5', monitor='val_iou_metrics', mode='max', save_best_only=True, save_weights_only=False, verbose=1)])


if __name__ == '__main__':
    train_and_predict()
