import os
from os.path import isfile, isdir, join, dirname
import pickle
from sklearn.model_selection import train_test_split


def image_preprocess(image):
    return normalize(image)


def normalize(image):
    return image/255.


def get_classes(cache_name):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        return cache['classes']

    return None


def create_training_instances(
    train_folder,
    valid_folder,
    cache_name
):
    train_entries = {}
    valid_entries = {}
    classes = []

    if cache_name and os.path.exists(cache_name):
        print('Loading data from .pkl file')
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        return cache['train'], cache['valid'], cache['classes']

    if not train_folder:
        print('Train folder is not set - exit')
        return train_entries, valid_entries, classes

    classes = [f for f in os.listdir(train_folder) if isdir(join(train_folder, f))]

    print(classes)

    train_cnt = 0

    for className in classes:
        class_dpath = join(train_folder, className)

        train_entries[className] = [join(class_dpath, f) for f in os.listdir(class_dpath) if isfile(join(class_dpath, f))]
        train_cnt += len(train_entries[className])

    print('Found {} train imgs'.format(train_cnt))

    valid_cnt = 0

    if not valid_folder:
        train_cnt = 0

        print('Validation folder is not set - Splitting train set to generate valid set')
        for key, value in train_entries.items():
            train_entries[key], valid_entries[key] = train_test_split(value, test_size=0.2, random_state=42)
            train_cnt += len(train_entries[key])
            valid_cnt += len(valid_entries[key])
    else:
        for className in classes:
            class_dpath = join(valid_folder, className)

            if not isdir(class_dpath):
                print('Class {} not exist'.format(className))

            valid_entries[className] = [join(class_dpath, f) for f in os.listdir(class_dpath)
                                        if isfile(join(class_dpath, f))]
            valid_cnt += len(valid_entries[className])

    print('Found {} train imgs'.format(train_cnt))
    print('Found {} valid imgs'.format(valid_cnt))

    if cache_name:
        if not isdir(dirname(cache_name)):
            os.makedirs(dirname(cache_name))

        cache = {'train': train_entries, 'valid': valid_entries, 'classes': classes}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_entries, valid_entries, classes


if __name__ == '__main__':
    train = '../data_root/__signs/robofest_data/signs_only'
    # valid = '../data_root/__signs/robofest_data/signs_only'
    create_training_instances(train, None, None)
