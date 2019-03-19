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
        all_insts = []
        seen_labels = {}
        
        for ann_dir in ann_dirs:
            for ann in sorted(os.listdir(ann_dir)):        
                
                img = {'object':[]}

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
                            print('Image file {} for {} not found'.format(elem.text, annot_file))
                            break;
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
                                    print('Ignore label: {}'.format(obj['name']))
                                    break
                                else:
                                    img['object'] += [obj]
                                    
                            if 'bndbox' in attr.tag:
                                for dim in list(attr):
                                    if 'xmin' in dim.tag:
                                        obj['xmin'] = int(round(float(dim.text)))
                                    if 'ymin' in dim.tag:
                                        obj['ymin'] = int(round(float(dim.text)))
                                    if 'xmax' in dim.tag:
                                        obj['xmax'] = int(round(float(dim.text)))
                                    if 'ymax' in dim.tag:
                                        obj['ymax'] = int(round(float(dim.text)))

                # if len(img['object']) > 0:
                if img is not None:
                    all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        if cache_name:
            with open(cache_name, 'wb') as handle:
                pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_insts, seen_labels


from sklearn.model_selection import train_test_split


def split_by_objects(instances, classes, rate):

    classes = {}

    for inst in instances:
        for obj in inst['object']:
            if obj['name'] in classes:
                classes[obj['name']] += [inst]
            else:
                classes[obj['name']] = [inst]

    train_entries = []
    valid_entries = []

    for _class, _imgs in classes.items():
        train_, valid_ = train_test_split(_imgs, test_size=rate, random_state=42)

        print('Splitted {} for {}/{}'.format(_class, len(train_), len(valid_)))

        train_entries += train_
        valid_entries += valid_

    print('Result split {}/{}'.format(len(train_entries), len(valid_entries)))

    return train_entries, valid_entries
