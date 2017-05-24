from __future__ import print_function

import os
import numpy as np
import cv2

data_path = '../positive'

npy_img_height = 240
npy_img_width = 320

def create_train_data():
    images = os.listdir(data_path)
    total = len(images)

    imgs = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, npy_img_height, npy_img_width), dtype=np.uint8)

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

        imgs_mask[i] = np.zeros((npy_img_height, npy_img_width), np.uint8)
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

if __name__ == '__main__':
    create_train_data()
