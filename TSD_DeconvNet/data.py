import os
from os.path import isfile, isdir, join, dirname, exists
import pickle
from sklearn.model_selection import train_test_split


def image_preprocess(image):
    return normalize(image)


def normalize(image):
    return image / 255.


def create_training_instances(
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        cache_name
):
    train_entries = []
    valid_entries = []

    if cache_name and os.path.exists(cache_name):
        print('Loading data from .pkl file')
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        return cache['train'], cache['valid']

    if not train_images or not train_masks:
        print('Train folder is not set - exit')
        return train_entries, valid_entries

    train_entries = [(join(train_images, f), join(train_masks, f))
                     for f in os.listdir(train_masks)
                     if isfile(join(train_masks, f)) and exists(join(train_images, f))]

    print('Found {} train imgs'.format(len(train_entries)))

    if not valid_images or not valid_masks:
        print('Validation folder is not set - Splitting train set to generate valid set')
        train_entries, valid_entries = train_test_split(train_entries, test_size=0.2, random_state=42)
    else:
        valid_entries = [(join(valid_images, f), join(valid_masks, f))
                         for f in os.listdir(valid_masks)
                         if isfile(join(valid_masks, f)) and exists(join(valid_images, f))]

    print('Found {} train imgs'.format(len(train_entries)))
    print('Found {} valid imgs'.format(len(valid_entries)))

    if cache_name:
        if not isdir(dirname(cache_name)):
            os.makedirs(dirname(cache_name))

        cache = {'train': train_entries, 'valid': valid_entries}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_entries, valid_entries


if __name__ == '__main__':
    train_im = '../../data_root/__signs/robofest_data/positive_bbox_class'
    train_msk = '../../data_root/__signs/robofest_data/positive_masks_train'
    valid_msk = '../../data_root/__signs/robofest_data/positive_masks_test'
    create_training_instances(train_im, train_msk, train_im, valid_msk, None)
