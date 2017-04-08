from __future__ import print_function

import os
import numpy as np
import cv2

data_path = 'raw/'

image_rows = 240
image_cols = 320

def create_train_data():
    images = os.listdir(data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        imgs[i] = cv2.imread(os.path.join(data_path, image_name))

        info = image_name.split(';')
        ul_x = max(0, int(info[1]))
        ul_y = max(0, int(info[2]))
        lr_x = ul_x + max(0, int(info[3]))
        lr_y = ul_y + max(0, int(info[4]))

        imgs_mask[i] = np.zeros((image_rows,image_cols), np.uint8)
        cv2.rectangle(imgs_mask[i], (ul_x, ul_y), (lr_x, lr_y), thickness=-1, color=255 )

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    # create_test_data()
