from sklearn.model_selection import train_test_split
import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle


def get_img_path(img_dirs, filename):
    for img_dir in img_dirs:
        filepath = os.path.join(img_dir, filename)

        if os.path.exists(filepath):
            return filepath

    return None


def parse_voc_annotation(ann_dirs, img_dirs, cache_name, labels=[]):
    # Dirty hack to check
    if cache_name and len(cache_name) > 1 and os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        if not isinstance(ann_dirs, list):
            ann_dirs = [ann_dirs]

        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]

        all_insts = []
        seen_labels = {}

        for ann_dir in ann_dirs:
            for ann in sorted(os.listdir(ann_dir)):

                img = {'object': []}

                try:
                    annot_file = os.path.join(ann_dir, ann)
                    tree = ET.parse(annot_file)
                except Exception as e:
                    print(e)
                    print('Ignore this bad annotation: ' + annot_file)
                    continue

                for elem in tree.iter():
                    if 'filename' in elem.tag:
                        fpath = get_img_path(img_dirs, elem.text)
                        if fpath is None:
                            img = None
                            print('Image file {} for {} not found'.format(
                                elem.text, annot_file))
                            break
                        else:
                            img['filename'] = fpath

                    if 'width' in elem.tag:
                        img['width'] = int(elem.text)
                    if 'height' in elem.tag:
                        img['height'] = int(elem.text)
                    if 'object' in elem.tag or 'part' in elem.tag:
                        obj = {}

                        for attr in list(elem):
                            if 'name' in attr.tag:
                                obj['name'] = attr.text

                                if obj['name'] in seen_labels:
                                    seen_labels[obj['name']] += 1
                                else:
                                    seen_labels[obj['name']] = 1

                                if len(labels) > 0 and obj['name'] not in labels:
                                    print('Ignore label: {}'.format(
                                        obj['name']))
                                    break
                                else:
                                    img['object'] += [obj]

                            if 'bndbox' in attr.tag:
                                for dim in list(attr):
                                    if 'xmin' in dim.tag:
                                        obj['xmin'] = int(
                                            round(float(dim.text)))
                                    if 'ymin' in dim.tag:
                                        obj['ymin'] = int(
                                            round(float(dim.text)))
                                    if 'xmax' in dim.tag:
                                        obj['xmax'] = int(
                                            round(float(dim.text)))
                                    if 'ymax' in dim.tag:
                                        obj['ymax'] = int(
                                            round(float(dim.text)))

#                 if len(img['object']) == 0:
#                     print('Warning! Zero objects on image {}'.format(img['filename']))

                if img is not None:
                    all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        if cache_name:
            with open(cache_name, 'wb') as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels


def replace_all_labels_2_one(instances, new_label):

    labels = {new_label: 0}

    for inst in instances:
        for obj in inst['object']:
            obj['name'] = new_label

            labels[new_label] += 1

    return instances, labels


def split_by_objects(instances, labels, rate):
    # TODO - Fix it! Not use now because it can put train samples to test =(
    classes = {}

    for inst in instances:
        for obj in inst['object']:
            if obj['name'] in classes:
                classes[obj['name']] += [inst]
            else:
                classes[obj['name']] = [inst]

    train_entries = []
    valid_entries = []

    train_lbls = {}
    valid_lbls = {}

    for _class, _imgs in classes.items():
        train_, valid_ = train_test_split(
            _imgs, test_size=rate, random_state=42)

        print('Splitted {} for {}/{}'.format(_class, len(train_), len(valid_)))

        train_entries += train_
        valid_entries += valid_

        train_lbls[_class] = len(train_)
        valid_lbls[_class] = len(valid_)

    print('Result split {}/{}'.format(len(train_entries), len(valid_entries)))

    return train_entries, train_lbls, valid_entries, valid_lbls

def get_labels_dict(instances):
    labels = {}

    for inst in instances:
        for obj in inst['object']:
            if obj['name'] in labels:
                labels[obj['name']] += 1
            else:
                labels[obj['name']] = 1

    return labels

def create_training_instances(
        train_annot_folder,
        train_image_folder,
        train_cache,
        valid_annot_folder,
        valid_image_folder,
        valid_cache,
        labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(
        train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if valid_annot_folder:
        valid_ints, valid_labels = parse_voc_annotation(
            valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        from sklearn.model_selection import train_test_split

        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_ints, valid_ints = train_test_split(train_ints,
                                                  test_size=0.3,
                                                  random_state=42)

        train_labels = get_labels_dict(train_ints)
        valid_labels = get_labels_dict(valid_ints)

        overlap_labels = set(train_labels.keys()).intersection(set(valid_labels.keys()))

        if len(overlap_labels) != len(train_labels.keys()) or \
           len(overlap_labels) != len(valid_labels.keys()):
            raise Exception('Invalid split of data: {} vs {}'.format(train_labels, valid_labels))

        # train_ints, train_labels, valid_ints, valid_labels = split_by_objects(
        # train_ints, train_labels, 0.2)

    print('After split: {} / {}'.format(len(train_ints), len(valid_ints)))

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t' + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) != len(labels):
            print(
                'Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        print(valid_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object'])
                             for inst in (train_ints + valid_ints)])

    return train_ints, train_labels, valid_ints, valid_labels, sorted(labels), max_box_per_image

