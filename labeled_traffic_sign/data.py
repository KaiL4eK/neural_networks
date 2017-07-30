from __future__ import print_function

import os
import numpy as np
import cv2
import net

raw_path  = ['../raw_data/car_register_video_annotated2' ]

data_path = '.'

npy_img_height = 240
npy_img_width = 320

module_delimiter        = '-'
module_data_delimiter   = ','
extension_delimiter     = '.'

id_module_idx           = 0
frame_number_module_idx = 1
bbox_module_idx         = 2
label_module_idx        = 3

label_list_scores       = [0] * len(net.glob_label_list)

def print_process(index, total = 0):
    if index % 50 == 0:
        if total != 0:
            print('Done: {}/{} images'.format(index, total))
        else:
            print('Done: {} images'.format(index))

def show_labels_score():
    print('Label score:')
    for i, label_score in enumerate(label_list_scores):
        print('\t%s: %d' % (net.glob_label_list[i], label_score))

def create_train_data():
    total = 0
    image_idx = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for raw_path_active in raw_path:
        dir_list = os.listdir(raw_path_active)
        for i in dir_list:
            i_path = os.path.join(raw_path_active, i)
            if os.path.isdir(i_path):
                total += len(os.listdir(i_path))

    print('Total images: %d' % total)

    imgs        = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_class  = np.ndarray((total, len(net.glob_label_list)),         dtype=np.uint8)

    for raw_path_active in raw_path:
        dir_list = os.listdir(raw_path_active)
        for i in dir_list:
            i_path = os.path.join(raw_path_active, i)
            if os.path.isdir(i_path):
                for image_name in os.listdir(i_path):
                    image_path = os.path.join(i_path, image_name)

                    if len(image_name.split(extension_delimiter)) != 2:
                        print('File extension not found')
                        exit(1)

                    info = image_name.split(extension_delimiter)[0]
                    module_list = info.split(module_delimiter)

                    if len(module_list) != 4:
                        print('Modules info broken')
                        exit(1)

                    module_label_list = module_list[label_module_idx].split(module_data_delimiter)
                    
                    # print('--------------------')
                    # print(module_label_vector)
                    # print(label_list)

                    module_label_vector = [0] * len(net.glob_label_list)
                    for label in module_label_list:
                        if label in net.glob_label_list:
                            label_list_idx = net.glob_label_list.index(label)
                            module_label_vector[label_list_idx] = 1
                            
                            label_list_scores[label_list_idx] += 1

                    # print(module_label_vector)
                    # print(label_list_scores)
                    # print(image_path)

                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)
                    
                    imgs[image_idx]         = image
                    imgs_class[image_idx]   = module_label_vector

                    image_idx += 1
                    print_process(image_idx, total)


    show_labels_score()

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
