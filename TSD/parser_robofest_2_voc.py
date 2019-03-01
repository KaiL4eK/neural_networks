import numpy as np
import cv2
import os
import tqdm
import glob
import csv

from lxml import etree as ET
from shutil import copyfile

import argparse
argparser = argparse.ArgumentParser(description='Help =)')
argparser.add_argument('-o', '--output', help='path to output dir')
argparser.add_argument('-i', '--input', help='path to input dir')
args = argparser.parse_args()


def main():
    input_dirpath = args.input
    output_dirpath = args.output

    image_paths = []

    for root, subdirs, files in os.walk(input_dirpath):
        pic_extensions = ('.png', '.PNG')
        image_paths += [os.path.join(root, file) for file in files if file.endswith(pic_extensions)]

    checked_files = {}

    annotation_train_fldr = os.path.join(output_dirpath, 'Annotations_train')
    annotation_test_fldr  = os.path.join(output_dirpath, 'Annotations_test')
    images_fldr = os.path.join(output_dirpath, 'Images')

    if not os.path.exists(annotation_train_fldr):
        os.makedirs(annotation_train_fldr)

    if not os.path.exists(annotation_test_fldr):
        os.makedirs(annotation_test_fldr)

    if not os.path.exists(images_fldr):
        os.makedirs(images_fldr)

    for image_fpath in image_paths:
        img = cv2.imread(image_fpath)
        img_height, img_width, channels = img.shape

        image_name = os.path.basename(image_fpath)
        # Get info about bbox
        info = image_name.split(';')
        id = info[0]
        xmin = max(0, int(info[1]))
        ymin = max(0, int(info[2]))
        width = max(0, int(info[3]))
        height = max(0, int(info[4]))
        class_name = info[5].split('.')[0]

        # Filter Stop signs
        if class_name == 'stop':
            continue

        xmax = xmin + width
        ymax = ymin + height
        width -= max(0, xmax - img_width + 1)
        height -= max(0, ymax - img_height + 1)

        xmax = xmin + width
        ymax = ymin + height

        info = (xmin, ymin, xmax, ymax, class_name)

        if image_name in checked_files:
            checked_files[image_name] += [info]
        else:
            checked_files[image_name] = [info]

        copyfile(image_fpath, os.path.join(images_fldr, image_name))

        # Flipped
        if class_name == 'forward and right':
            class_name = 'forward and left'
        elif class_name == 'right':
            class_name = 'left'
        elif class_name == 'left':
            class_name = 'right'

        xmin = img_width - xmax
        xmax = xmin + width

        img = cv2.flip(img, 1)

        image_name = '{};{};{};{};{};{}.png'.format(id+'m', xmin, ymin, width, height, class_name)
        cv2.imwrite(os.path.join(images_fldr, image_name), img)

        info = (xmin, ymin, xmax, ymax, class_name)
        if image_name in checked_files:
            checked_files[image_name] += [info]
        else:
            checked_files[image_name] = [info]

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
                cv2.waitKey(100)

        tree = ET.ElementTree(xml_root)
        tree.write(os.path.join(annotation_train_fldr, file.split('.')[0] + '.xml'), pretty_print=True, xml_declaration=True,   encoding="utf-8")


if __name__ == '__main__':
    main()

