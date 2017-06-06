from __future__ import print_function

import os
import numpy as np
import cv2

data_path = '.'

raw_path  = ['../raw_data/lane_with_masks/training' ]
negative_path = [ '../raw_data/negative' ]

negatives_append = True

npy_img_height = 480
npy_img_width = 640

def print_process(index, total):
    if index % 100 == 0:
        print('Done: {0}/{1} images'.format(index, total))

def create_train_data():
    total = 0
    i = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    total = 95  # Sorry for hard code =)
    # for raw_path_active in raw_path:
    #     img_list = os.listdir(raw_path_active + '/image_2')
    #     total += len(img_list)

    if negatives_append:
        for neg_path in negative_path:
            img_list = os.listdir(neg_path)
            total += len(img_list)

    imgs        = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_mask   = np.ndarray((total, npy_img_height, npy_img_width), dtype=np.uint8)

    print(total)

    for raw_path_active in raw_path:
        image_path = raw_path_active + '/image_2'
        gt_path = raw_path_active + '/gt_image_2'
        print('Processing path: {}'.format(image_path))
        images = os.listdir(image_path)

        for image_name in images:
            basename = image_name.split('.')[0]

            prefix = basename.split('_')[0]
            number = basename.split('_')[1]
            if prefix == 'um':
                gt_name = prefix + '_lane_' + number + '.png'
            else:
                continue

            img = cv2.imread(os.path.join(image_path, image_name))
            img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_CUBIC)

            ref = cv2.imread(os.path.join(gt_path, gt_name))
            ref = cv2.inRange(ref, (255, 0, 255), (255, 0, 255))
            ref = cv2.resize(ref, (npy_img_width, npy_img_height), interpolation = cv2.INTER_NEAREST)
            
            # res = cv2.bitwise_and(img, img, mask=ref)
            # cv2.imshow('1', img)
            # cv2.imshow('2', ref)
            # cv2.imshow('3', res)
            # if cv2.waitKey(0) == ord('q'):
                # exit(0)

            imgs[i]         = img
            imgs_mask[i]    = ref

            print_process(i, total)
            i += 1
            
    if negatives_append:
        for neg_path in negative_path:
            print('Processing negatives path: {}'.format(neg_path))
            images = os.listdir(neg_path)

            for image_name in images:
                img = cv2.imread(os.path.join(neg_path, image_name))
                img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_CUBIC)

                imgs[i]         = img
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
