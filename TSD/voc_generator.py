import tqdm
import cv2
import os
from lxml import etree as ET
from shutil import copyfile



def copy_2_images_dir(root_dir, src_fpaths):
    imgs_directory = os.path.join(root_dir, 'Images')
    if not os.path.exists(imgs_directory):
        os.makedirs(imgs_directory)

    for frame_fpath in tqdm.tqdm(src_fpaths):
        dst_fpath = os.path.join(imgs_directory, os.path.basename(frame_fpath))
        if not os.path.exists(dst_fpath):
            copyfile(frame_fpath, dst_fpath)


def generate_annotations(root_dir, checked_files, show_frames=False):
    ants_directory = os.path.join(root_dir, 'Annotations')
    if not os.path.exists(ants_directory):
        os.makedirs(ants_directory)

    imgs_directory = os.path.join(root_dir, 'Images')

    for fname in tqdm.tqdm(checked_files.keys()):
        img_fpath = os.path.join(imgs_directory, fname)
        # print('Reading {}'.format(img_fpath))
        img_h, img_w, img_c = cv2.imread(img_fpath).shape

        xml_root = ET.Element("annotation")
        xml_filename = ET.SubElement(xml_root, "filename")
        xml_filename.text = fname

        xml_size = ET.SubElement(xml_root, "size")
        xml_width = ET.SubElement(xml_size, "width")
        xml_width.text = str(img_w)
        xml_height = ET.SubElement(xml_size, "height")
        xml_height.text = str(img_h)
        xml_depth = ET.SubElement(xml_size, "depth")
        xml_depth.text = str(img_c)

        infos = checked_files[fname]
        if infos is not None:
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

                if show_frames:
                    cv2.rectangle(img, (xmin, ymin),
                                  (xmax, ymax), (255, 0, 0), 3)

                    cv2.imshow('Result', img)
                    cv2.waitKey(1000)

        tree = ET.ElementTree(xml_root)
        tree.write(os.path.join(ants_directory, fname.split('.')[0] + '.xml'),
                   pretty_print=True,
                   xml_declaration=True,
                   encoding="utf-8")
