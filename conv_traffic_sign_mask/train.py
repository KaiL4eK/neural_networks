from __future__ import print_function

import os
import cv2
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
import random
import itertools

from data import *
from net import *
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-b', '--batch_size', default=1, action='store', help='Size of batch to learn')
parser.add_argument('-a', '--augmentation', action='store_true', help='Use augmentation')

args = parser.parse_args()

def preprocess_arrays(imgs, masks):
    imgs_p       = np.ndarray((imgs.shape[0],  nn_img_h, nn_img_w, 3),            dtype=np.float32)
    masks_p      = np.ndarray((masks.shape[0], nn_out_h, nn_out_w),               dtype=np.float32)
    grid_masks_p = np.ndarray((masks.shape[0], nn_grid_y_count, nn_grid_x_count), dtype=np.float32)

    for i in range(masks_p.shape[0]):
        imgs_p[i]       = preprocess_img(imgs[i])
        masks_p[i]      = preprocess_mask(masks[i])
        # grid_masks_p[i] = get_grid_mask(masks_p[i])

    # p = np.random.permutation(len(imgs_p))
    # imgs_p        = imgs_p[p]
    # masks_p       = masks_p[p]
    # grid_masks_p  = grid_masks_p[p]

    return imgs_p, masks_p[..., np.newaxis], grid_masks_p[..., np.newaxis]

def get_grid_mask(mask):
    if mask.shape[0] != nn_img_h or mask.shape[1] != nn_img_w:
        print('Incorrect shape of input mask [%s()]' % (get_grid_mask.__name__))
        exit(1)

    grid_mask = np.ndarray((nn_grid_y_count, nn_grid_x_count), dtype=np.float32)
    for y in range(nn_grid_y_count):
        for x in range(nn_grid_x_count):
            x_px, y_px = (x*nn_grid_cell_size, y*nn_grid_cell_size)
            img_part = mask[y_px : y_px + nn_grid_cell_size, x_px : x_px + nn_grid_cell_size]

            if np.sum(img_part) > 0:
                grid_mask[y, x] = 1.
            else:
                grid_mask[y, x] = 0

    if 0:
        show_mask_g = cv2.resize(grid_mask, (600, 300), interpolation = cv2.INTER_NEAREST)
        # show_mask_r = cv2.resize(mask,       (600, 300), interpolation = cv2.INTER_NEAREST)

        # show_mask_g = cv2.cvtColor(show_mask_g, cv2.COLOR_GRAY2BGR)
        # show_mask_g[np.where((show_mask_g == (0, 0, 0)).all(axis=2))] = (0,0,255)
        # show_mask_g[np.where((show_mask_r != 0))] = (255,0,0)

        cv2.imshow('1', show_mask_g)
        cv2.imshow('2', cv2.resize(mask, (600, 300), interpolation = cv2.INTER_NEAREST))
        # cv2.imshow('3', show_mask_r)
        # cv2.imwrite('test.png', imgs[i])
        if cv2.waitKey(0) == ord('q'):
            exit(1)

    return grid_mask

def grid_generator(augmen_generator):
    image_idx = 0
    for image_batch, mask_batch in augmen_generator:
        grid_mask_batch = np.ndarray((mask_batch.shape[0], nn_grid_y_count, nn_grid_x_count), dtype=np.float32)

        for i, mask in enumerate(mask_batch):          
            grid_mask_batch[i] = get_grid_mask(mask)

            # cv2.imwrite('augment'+'/img_'+str(image_idx)+'.png', (image_batch[i] * 255).astype(np.uint8))
            # cv2.imwrite('augment'+'/gmask_'+str(image_idx)+'.png', cv2.resize((grid_mask_batch[i] * 255).astype(np.uint8), (nn_img_w, nn_img_h), interpolation = cv2.INTER_NEAREST))
            
            image_idx += 1

            # print(image_batch[i].shape)

        # cv2.imshow('1', cv2.resize(image_batch[0], (600, 300), interpolation = cv2.INTER_LINEAR))
        # cv2.imshow('2', cv2.resize(grid_mask_batch[i], (600, 300), interpolation = cv2.INTER_NEAREST))
        # if cv2.waitKey(0) == ord('q'):
            # exit(1)

        yield image_batch, grid_mask_batch[..., np.newaxis]


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train      = npy_data_load_images()
    imgs_mask_train = npy_data_load_masks()
    imgs_train, imgs_mask_train, imgs_grid_mask_train = preprocess_arrays(imgs_train, imgs_mask_train)

    if len( np.unique(imgs_mask_train) ) > 2:
        print('Preprocessing created mask with more than two binary values')
        exit(1)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = get_unet(lr=1e-4)
    batch_size = int(args.batch_size)

    print('Batch size is set to %d' % batch_size)

    if args.weights:
        model.load_weights(args.weights)

    if args.augmentation:
        data_gen_args = dict( rotation_range=10,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              fill_mode='constant',
                              cval=0 )

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        seed = 1
        # 'Only required if featurewise_center or featurewise_std_normalization or zca_whitening.'
        # image_datagen.fit(imgs_agm, augment=True, seed=seed)
        # mask_datagen.fit(imgs_mask_agm, augment=True, seed=seed)

        # save_to_dir=augment_path
        image_generator = image_datagen.flow(imgs_train, batch_size=batch_size, save_prefix='img', seed=seed)
        mask_generator  = mask_datagen.flow(imgs_mask_train, batch_size=batch_size, save_prefix='mask', seed=seed)

        train_generator = itertools.izip(image_generator, mask_generator)

        model.fit_generator(train_generator, steps_per_epoch=len(imgs_train) / batch_size, 
            epochs=70000, validation_data=(imgs_train, imgs_mask_train),
            callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

    else:

        print('-'*30)
        print('Fitting model...')
        print('-'*30)

        model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=7000, verbose=1, shuffle=True, validation_split=0.1, 
            callbacks=[ModelCheckpoint('weights_best.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)])

if __name__ == '__main__':
    train_and_predict()
