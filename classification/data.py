from __future__ import print_function

import os
import numpy as np
import cv2

raw_path  = ['../raw_data/positive_bbox_class' ]
negative_path = [ '../raw_data/negative' ]

all_paths = raw_path + negative_path

data_path = '.'

npy_img_height = 240
npy_img_width = 320

# Bbox is compiled as [ul_x, ul_y, w, h]
check_data = False

def print_process(index, total):
    if index % 100 == 0:
        print('Done: {0}/{1} images'.format(index, total))

def create_train_data():
    total   = 0
    i       = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for path in all_paths:
        print(path)
        img_list = os.listdir(path)
        total += len(img_list)

    imgs        = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_class  = np.ndarray((total, ), dtype=object)

    for raw_path_active in raw_path:
        print('Processing path: {}'.format(raw_path_active))
        images = os.listdir(raw_path_active)

        for image_name in images:
            # Save image
            img = cv2.imread(os.path.join(raw_path_active, image_name))

            info   = image_name.split(';')
            if len(info) > 2:
                name   = info[5].split('.')[0]
            else:
                name   = info[1].split('.')[0]

            img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)

            imgs[i]         = np.array([img])
            imgs_class[i]   = name

            # print(imgs_class[i])

            print_process(i, total)
            i += 1

    for neg_path in negative_path:
        print('Processing negatives path: {}'.format(neg_path))
        images = os.listdir(neg_path)

        for image_name in images:
            # Save image
            img = cv2.imread(os.path.join(neg_path, image_name))
            img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)

            imgs[i]         = img
            imgs_class[i]   = 'no_sign'

            print_process(i, total)
            i += 1

    np.save(data_path + '/imgs_train.npy', imgs)
    np.save(data_path + '/imgs_class_train.npy', imgs_class)
    print('Loading done.')
    print('Saving to .npy files done.')

def npy_data_load_images():
    return np.load(data_path + '/imgs_train.npy')

def npy_data_load_classes():
    return np.load(data_path + '/imgs_class_train.npy')

if __name__ == '__main__':
    create_train_data()
