import os
import numpy as np
import cv2

data_root = '../data_root/__signs/robofest_data/signs_only'

save_data_path = 'data'

npy_imgs_filename = 'imgs_train.npy'
npy_lbls_filename = 'lbls_train.npy'

npy_img_height = 48
npy_img_width = 48

sum_0 = 0
sum_1 = 0


def normalize(image):
    return image/255.


def create_training_instances(
    train_folder,
    valid_folder,
    cache_name
):
    from os.path import isfile, isdir, join, dirname
    import pickle

    from sklearn.model_selection import train_test_split

    train_entries = {}
    valid_entries = {}
    classes = []

    if os.path.exists(cache_name):
        print('Loading data from .pkl file')
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        return cache['train'], cache['valid'], cache['classes']

    if not train_folder:
        print('Train folder is not set - exit')
        return train_entries, valid_entries, classes

    classes = [f for f in os.listdir(train_folder) if isdir(join(train_folder, f))]

    print(classes)

    for className in classes:
        class_dpath = join(train_folder, className)

        train_entries[className] = [join(class_dpath, f) for f in os.listdir(class_dpath) if isfile(join(class_dpath, f))]

    if not valid_folder:
        print('Validation folder is not set - Splitting train set to generate valid set')
        for key, value in train_entries.items():
            train_entries[key], valid_entries[key] = train_test_split(value, test_size=0.2, random_state=42)
    else:
        for className in classes:
            class_dpath = join(valid_folder, className)

            if not isdir(class_dpath):
                print('Class {} not exist'.format(className))

            valid_entries[className] = [join(class_dpath, f) for f in os.listdir(class_dpath)
                                        if isfile(join(class_dpath, f))]

    if cache_name:
        if not isdir(dirname(cache_name)):
            os.makedirs(dirname(cache_name))

        cache = {'train': train_entries, 'valid': valid_entries, 'classes': classes}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_entries, valid_entries, classes



def create_train_data():
    import readTrafficSigns as ts

    images, labels = ts.readTrafficSigns(data_root)

    total = len(images)

    npy_img_list = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    npy_lbl_list = np.ndarray((total), dtype=np.uint8)

    for i, img in enumerate(images):
        # (h, w, c)
        # print(img.shape)

        c_img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation=cv2.INTER_LINEAR)

        # Conversion just for OpenCV rendering
        # c_img = cv2.cvtColor(c_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('res', c_img)
        # cv2.waitKey(1000)

        npy_img_list[i] = c_img
        npy_lbl_list[i] = labels[i]

    # sum_0 += img.shape[0]
    # sum_1 += img.shape[1]

    # print( sum_0 / len(images), sum_1 / len(images) )

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    np.save(os.path.join(save_data_path, npy_imgs_filename), npy_img_list)
    np.save(os.path.join(save_data_path, npy_lbls_filename), npy_lbl_list)

    print('Saving to .npy {} files done'.format(total))


def npy_data_load_images():
    return np.load(os.path.join(save_data_path, npy_imgs_filename))


def npy_data_load_labels():
    return np.load(os.path.join(save_data_path, npy_lbls_filename))


if __name__ == '__main__':
    train = '../data_root/__signs/robofest_data/signs_only'
    # valid = '../data_root/__signs/robofest_data/signs_only'
    create_training_instances(train, None, None)
