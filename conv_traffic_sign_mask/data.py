from __future__ import print_function

import os
import numpy as np
import cv2

data_path = '.'

raw_path  = ['../raw_data/car_register_video_annotated1' ]
negative_path = [ '../raw_data/negative' ]

negatives_append = False

npy_img_height = 480
npy_img_width = 640

def print_process(index, total = 0):
    if index % 50 == 0:
        if total != 0:
            print('Done: {}/{} images'.format(index, total))
        else:
            print('Done: {} images'.format(index))

def create_train_data():
    total = 0
    i = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    total = 0

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
