import numpy as np
import cv2
import os
import tqdm
import glob
import csv

from lxml import etree as ET
from shutil import copyfile

data_root_path = 'RTSD'

frames_paths = ['rtsd-d1-frames']
gt_frame_fpaths = []

for frames_path in frames_paths:
    frames_dirpath = os.path.join(data_root_path, frames_path)

    for subdir in ['test', 'train']:
        path = os.path.join(frames_dirpath, subdir)

        gt_frame_fpaths += [os.path.join(path, i) for i in os.listdir(path)]


gt_paths = ['rtsd-d1-gt']
gt_classes = []

gt_train_csv_fname = 'train_gt.csv'
gt_test_csv_fname  = 'test_gt.csv'

gt_train_annot_fpaths = []
gt_test_annot_fpaths  = []

for gt_path in gt_paths:
    gt_dirpath = os.path.join(data_root_path, gt_path)
    local_classes = os.listdir(gt_dirpath)

    for local_class in local_classes:
        class_dirpath = os.path.join(gt_dirpath, local_class)

        if os.path.isdir(class_dirpath):
            if local_class not in gt_classes:
                gt_classes.append(local_class)

            gt_train_annot_fpaths.append((local_class, os.path.join(class_dirpath, gt_train_csv_fname)))
            gt_test_annot_fpaths.append((local_class, os.path.join(class_dirpath, gt_test_csv_fname)))

print(gt_train_annot_fpaths)
print(gt_test_annot_fpaths)
print(gt_classes)



dst_data_dir = 'RTSD_voc'
annotation_train_fldr = os.path.join(dst_data_dir, 'Annotations_train')
annotation_test_fldr  = os.path.join(dst_data_dir, 'Annotations_test')
images_fldr = os.path.join(dst_data_dir, 'Images')

if not os.path.exists(annotation_train_fldr):
    os.makedirs(annotation_train_fldr)

if not os.path.exists(annotation_test_fldr):
    os.makedirs(annotation_test_fldr)

if not os.path.exists(images_fldr):
    os.makedirs(images_fldr)

print('Copy frames to dst folder')

for frame_fpath in tqdm.tqdm(gt_frame_fpaths):
    copyfile(frame_fpath, os.path.join(images_fldr, os.path.basename(frame_fpath)))

FILENAME_KEY='filename'
LEFT_X_KEY  ='x_from'
UUPPER_Y_KEY='y_from'
WIDTH_KEY   ='width'
HEIGHT_KEY  ='height'
SIGN_CLS_KEY='sign_class'

checked_files = {}

print('Reading annotations')

for train_annot in tqdm.tqdm(gt_train_annot_fpaths):
    class_name, annot_fpath = train_annot

    with open(annot_fpath, 'r') as annot_fd:
        d_reader = csv.DictReader(annot_fd)

        for row in d_reader:
            fname = row[FILENAME_KEY]
            xmin = int(row[LEFT_X_KEY])
            ymin = int(row[UUPPER_Y_KEY])
            xmax = int(xmin) + int(row[WIDTH_KEY])
            ymax = int(ymin) + int(row[HEIGHT_KEY])

            if fname in checked_files:
                checked_files[fname] += [(xmin, ymin, xmax, ymax, class_name)]
            else:
                checked_files[fname] = [(xmin, ymin, xmax, ymax, class_name)]

print('Creating annotations')

for file in tqdm.tqdm(checked_files.keys()):

    img = cv2.imread(os.path.join(images_fldr, file))

    xml_root = ET.Element("annotation")
    xml_filename = ET.SubElement(xml_root, "filename")
    xml_filename.text = file

    xml_size = ET.SubElement(xml_root, "size")
    xml_width = ET.SubElement(xml_size, "width")
    xml_width.text = str(img.shape[1])
    xml_height = ET.SubElement(xml_size, "height")
    xml_height.text = str(img.shape[0])
    xml_depth = ET.SubElement(xml_size, "depth")
    xml_depth.text = str(img.shape[2])

    if file in checked_files:
        infos = checked_files[file]

        for info in infos:

            xmin, ymin, xmax, ymax, class_name = info

            xml_object = ET.SubElement(xml_root, "object")
            xml_name = ET.SubElement(xml_object, "name")
            xml_name.text = class_name

            xml_bndbox = ET.SubElement(xml_object, "bndbox")

            xml_xmin = ET.SubElement(xml_bndbox, "xmin")
            xml_xmin.text = str(xmin)
            xml_ymin = ET.SubElement(xml_bndbox, "ymin")
            xml_ymin.text = str(ymin)
            xml_xmax = ET.SubElement(xml_bndbox, "xmax")
            xml_xmax.text = str(xmax)
            xml_ymax = ET.SubElement(xml_bndbox, "ymax")
            xml_ymax.text = str(ymax)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 3)

            cv2.imshow('Result', img)
            cv2.waitKey(1000)

    tree = ET.ElementTree(xml_root)
    tree.write(os.path.join(annotation_train_fldr, file.split('.')[0] + '.xml'), pretty_print=True, xml_declaration=True,   encoding="utf-8")
