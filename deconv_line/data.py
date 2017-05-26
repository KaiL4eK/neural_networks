from __future__ import print_function

import os
import numpy as np
import cv2

raw_path  = ['../raw_data/positive_bbox_class' ]
negative_path = [ '../raw_data/negative' ]

all_paths = raw_path + negative_path

npy_img_height = 240
npy_img_width = 240

def print_process(index, total):
    if index % 100 == 0:
        print('Done: {0}/{1} images'.format(index, total))

def create_train_data():
    total = 0
    i = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for path in all_paths:
        print(path)
        img_list = os.listdir(path)
        total += len(img_list)

    imgs        = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_mask   = np.ndarray((total, npy_img_height, npy_img_width), dtype=np.uint8)

    for raw_path_active in raw_path:
        print('Processing path: {}'.format(raw_path_active))
        images = os.listdir(raw_path_active)

        for image_name in images:
            # Save image
            img = cv2.imread(os.path.join(raw_path_active, image_name))
            img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)

            # Get info about bbox
            img_height, img_width, channels = img.shape
            scale_x = float(npy_img_width)/img_width
            scale_y = float(npy_img_height)/img_height

            info   = image_name.split(';')
            ul_x   = max(0, int(float(info[1]) * scale_x))
            ul_y   = max(0, int(float(info[2]) * scale_y))
            width  = max(0, int(float(info[3]) * scale_x))
            height = max(0, int(float(info[4]) * scale_y))
            name   = info[5].split('.')[0]

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

            mask = np.zeros((npy_img_height, npy_img_width), dtype=np.uint8)
            cv2.rectangle(mask, (ul_x, ul_y), (lr_x, lr_y), thickness=-1, color=255 )

            imgs[i]         = np.array([img])
            imgs_mask[i]    = mask

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

            imgs[i]         = np.array([img])
            imgs_mask[i]    = np.zeros((npy_img_height, npy_img_width), dtype=np.uint8)

            print_process(i, total)
            i += 1

    np.save(data_path + '/imgs_train.npy', imgs)
    np.save(data_path + '/imgs_mask_train.npy', imgs_mask)
    print('Loading done.')
    print('Saving to .npy files done.')

def npy_data_load_images():
    return np.load(data_path + '/imgs_train.npy')

def npy_data_load_masks():
    return np.load(data_path + '/imgs_mask_train.npy')

if __name__ == '__main__':
    create_train_data()