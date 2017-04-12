from __future__ import print_function

import os
import numpy as np
import cv2

data_path = '../raw/'

npy_img_height = 240
npy_img_width = 320

# Bbox is compiled as [ul_x, ul_y, w, h]
check_data = False

def create_train_data():
    images = os.listdir(data_path)
    total = len(images)

    imgs = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_bbox = np.ndarray((total, 4), dtype=np.uint16)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        img = cv2.imread(os.path.join(data_path, image_name))
        img_height, img_width, channels = img.shape
        imgs[i] = np.array([img])

        scale_x = float(npy_img_width)/img_width
        scale_y = float(npy_img_height)/img_height

        info   = image_name.split(';')
        ul_x   = max(0, int(float(info[1]) * scale_x))
        ul_y   = max(0, int(float(info[2]) * scale_y))
        width  = max(0, int(float(info[3]) * scale_x))
        height = max(0, int(float(info[4]) * scale_y))

        lr_x = ul_x + width
        lr_y = ul_y + height
        width  -= max(0, lr_x - img_width + 1)
        height -= max(0, lr_y - img_height + 1)

        lr_x = ul_x + width
        lr_y = ul_y + height

        if check_data:
            if lr_x >= img_width:
                print('Warning1 {}, {}!'.format(img_width, lr_x))
            if lr_y >= img_height:
                print('Warning2 {}, {}!'.format(img_height, lr_y))

        # cv2.rectangle(img, (ul_x, ul_y), (lr_x, lr_y), thickness=3, color=(255, 0, 0) )
        # cv2.imshow('1', img)
        # cv2.waitKey(0)

        imgs_bbox[i] = np.array([ul_x, ul_y, width, height])

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_bbox_train.npy', imgs_bbox)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_bbox_train = np.load('imgs_bbox_train.npy')
    return imgs_train, imgs_bbox_train

if __name__ == '__main__':
    create_train_data()
