from __future__ import print_function

import os
import numpy as np
import cv2

import argparse

data_path = '.'

raw_path  = ['../raw_data/car_register_video_annotated1' ]
augment_path = 'augment'

negatives_append = False

npy_img_height = 800
npy_img_width = 1600

def print_process(index, total = 0):
    if index % 50 == 0:
        if total != 0:
            print('Done: {}/{} images'.format(index, total))
        else:
            print('Done: {} images'.format(index))

def process_augmented_images():
    augmented_images_masks = os.listdir(augment_path)
    total_agms = len(augmented_images_masks) / 2

    print('Counted %d augmented frames' % (total_agms))

    image_idx = 0

    imgs        = np.ndarray((total_agms, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_mask   = np.ndarray((total_agms, npy_img_height, npy_img_width), dtype=np.uint8)

    for file in augmented_images_masks:
        if file.startswith('mask') or file.startswith('.'):
            continue

        frame = cv2.imread(os.path.join(augment_path, file))
        mask = cv2.imread(os.path.join(augment_path, file.replace('img', 'mask')), cv2.IMREAD_GRAYSCALE)

        # print('Reading %s with %s' % (file, file.replace('img', 'mask'))) 
        if frame is None or mask is None:
            print('Skip image')
            continue

        imgs[image_idx]         = frame
        imgs_mask[image_idx]    = mask

        image_idx += 1
        print_process(image_idx, total_agms)

    print('Done %d images' % image_idx)
    return imgs[0:image_idx], imgs_mask[0:image_idx]


def execute_augmentation(imgs, imgs_mask):
    print('-'*30)
    print('Start augmentation...')
    print('-'*30)

    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy
    from keras.preprocessing.image import ImageDataGenerator
    import itertools

    imgs_mask_agm   = imgs_mask[..., np.newaxis]
    imgs_agm        = imgs
    for i in range(len(imgs_agm)):
        imgs_agm[i] = cv2.cvtColor(imgs_agm[i], cv2.COLOR_BGR2RGB)

    model = Sequential()
    model.add(Conv2D(1, (9, 9), activation='hard_sigmoid', padding='same', input_shape=(npy_img_height, npy_img_width, 3)))
    model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[])

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

    image_generator = image_datagen.flow(imgs_agm, batch_size=1, save_to_dir=augment_path, save_prefix='img', seed=seed)
    mask_generator  = mask_datagen.flow(imgs_mask_agm, batch_size=1, save_to_dir=augment_path, save_prefix='mask', seed=seed)

    train_generator = itertools.izip(image_generator, mask_generator)

    total_agms = 10
    image_idx = 0
    for batch in train_generator:
        batch_X, batch_Y = batch

        image_idx += 1
        print_process(image_idx, total_agms)

        if image_idx == total_agms:
            break

def create_train_data():
    parser = argparse.ArgumentParser(description='Process video with ANN')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Use augmented images')
    parser.add_argument('-p', '--process_augmented', action='store_true', help='Use augmented images')

    args = parser.parse_args()

    if not args.process_augmented:
        total = 0

        print('-'*30)
        print('Creating training images...')
        print('-'*30)

        for raw_path_active in raw_path:
            dir_list = os.listdir(raw_path_active)
            for i in dir_list:
                i_path = os.path.join(raw_path_active, i)
                if os.path.isdir(i_path):
                    # print(i_path)
                    image_dir_list = os.listdir(i_path)
                    for image_name in image_dir_list:
                        if len(image_name.split('.')) == 2:
                            frame_name = image_name.split('.')[0]
                            frame_path = os.path.join(i_path, frame_name)
                            mask_path = frame_path + '.mask.0.png'

                            if os.path.exists(mask_path):
                                total += 1


        print('Counted %d masks' % (total))

        image_idx = 0 

        imgs        = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
        imgs_mask   = np.ndarray((total, npy_img_height, npy_img_width), dtype=np.uint8)

        for raw_path_active in raw_path:
            dir_list = os.listdir(raw_path_active)
            for i in dir_list:
                i_path = os.path.join(raw_path_active, i)
                if os.path.isdir(i_path):
                    # print(i_path)
                    image_dir_list = os.listdir(i_path)
                    for image_name in image_dir_list:
                        if len(image_name.split('.')) == 2:
                            frame_name = image_name.split('.')[0]
                            frame_path = os.path.join(i_path, frame_name)
                            mask_path = frame_path + '.mask.0.png'
                            frame_path = frame_path + '.png'

                            if os.path.exists(mask_path):
                                # print(mask_path)

                                frame = cv2.imread(frame_path)
                                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame = cv2.resize(frame, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)

                                mask  = cv2.imread(mask_path)
                                mask  = cv2.resize(mask, (npy_img_width, npy_img_height), interpolation = cv2.INTER_NEAREST)

                                mask = cv2.inRange(mask, (0, 0, 127), (255, 255, 255)) 

                                # frame_2_show = np.copy(frame)
                                # frame_2_show[np.where((mask != 0))] = (0,255,0)
                                # frame_2_show = np.hstack(( cv2.resize(frame_2_show, (640, 480)), cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (640, 480)) ))

                                # cv2.imshow('1', mask);
                                # if cv2.waitKey(1) == ord('q'):
                                    # exit(1)

                                # print(np.unique(mask))

                                imgs[image_idx]         = frame
                                imgs_mask[image_idx]    = mask

                                image_idx += 1
                                print_process(image_idx, total)

        if args.augmentation:
            execute_augmentation(imgs, imgs_mask)
            
            imgs, imgs_mask = process_augmented_images()
    else:
        imgs, imgs_mask = process_augmented_images()

    np.save(data_path + '/imgs_train.npy', imgs)
    np.save(data_path + '/imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')

def npy_data_load_images():
    return np.load(data_path + '/imgs_train.npy')

def npy_data_load_masks():
    return np.load(data_path + '/imgs_mask_train.npy')

if __name__ == '__main__':
    create_train_data()
