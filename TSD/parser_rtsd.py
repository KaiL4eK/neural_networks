import numpy as np
import cv2
import os
import tqdm
import glob

from lxml import etree as ET
from shutil import copyfile

data_root_path = '../data_root/GTSDB_orig/FullIJCNN2013'


imgs_path = data_root_path
annots_fpath = data_root_path + '/gt.txt'

dst_data_dir = 'GTSDB_voc'
annotation_fldr = dst_data_dir + '/Annotations'
images_fldr = dst_data_dir + '/Images'

class_dict = {
    0: 'speed limit 20 (prohibitory)',
    1: 'speed limit 30 (prohibitory)',
    2: 'speed limit 50 (prohibitory)',
    3: 'speed limit 60 (prohibitory)',
    4: 'speed limit 70 (prohibitory)',
    5: 'speed limit 80 (prohibitory)',
    6: 'restriction ends 80 (other)',
    7: 'speed limit 100 (prohibitory)',
    8: 'speed limit 120 (prohibitory)',
    9: 'no overtaking (prohibitory)',
    10: 'no overtaking (trucks) (prohibitory)',
    11: 'priority at next intersection (danger)',
    12: 'priority road (other)',
    13: 'give way (other)',
    14: 'stop (other)',
    15: 'no traffic both ways (prohibitory)',
    16: 'no trucks (prohibitory)',
    17: 'no entry (other)',
    18: 'danger (danger)',
    19: 'bend left (danger)',
    20: 'bend right (danger)',
    21: 'bend (danger)',
    22: 'uneven road (danger)',
    23: 'slippery road (danger)',
    24: 'road narrows (danger)',
    25: 'construction (danger)',
    26: 'traffic signal (danger)',
    27: 'pedestrian crossing (danger)',
    28: 'school crossing (danger)',
    29: 'cycles crossing (danger)',
    30: 'snow (danger)',
    31: 'animals (danger)',
    32: 'restriction ends (other)',
    33: 'go right (mandatory)',
    34: 'go left (mandatory)',
    35: 'go straight (mandatory)',
    36: 'go right or straight (mandatory)',
    37: 'go left or straight (mandatory)',
    38: 'keep right (mandatory)',
    39: 'keep left (mandatory)',
    40: 'roundabout (mandatory)',
    41: 'restriction ends (overtaking) (other)',
    42: 'restriction ends (overtaking (trucks)) (other)',
}

if not os.path.exists(annotation_fldr):
    os.makedirs(annotation_fldr)

if not os.path.exists(images_fldr):
    os.makedirs(images_fldr)

checked_files = {}

with open(annots_fpath) as fp: 
    for cnt, line in enumerate(fp):
        fname = line.split(';')[0]
        xmin = int(line.split(';')[1])
        ymin = int(line.split(';')[2])
        xmax = int(line.split(';')[3])
        ymax = int(line.split(';')[4])
        classId = int(line.split(';')[5])

        if fname in checked_files:
            checked_files[fname] += [[xmin, ymin, xmax, ymax, classId]]
        else:
            checked_files[fname] = [[xmin, ymin, xmax, ymax, classId]]


# print("Line {}: {} / {} / {} / {} / {} / {}".format(cnt, fname, xmin, ymin, xmax, ymax, classId))
# print(checked_files)


for fpath in tqdm.tqdm(glob.glob(imgs_path + '/*.ppm')):
    
    file = os.path.basename(fpath)

# for file, infos in tqdm.tqdm(checked_files.items()):

    img = cv2.imread(os.path.join(imgs_path, file))

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
        # print(file)
        for info in infos:

            xmin, ymin, xmax, ymax, classId = info

            xml_object = ET.SubElement(xml_root, "object")
            xml_name = ET.SubElement(xml_object, "name")
            # xml_name.text = str(classId)
            xml_name.text = 'sign'

            xml_bndbox = ET.SubElement(xml_object, "bndbox")

            xml_xmin = ET.SubElement(xml_bndbox, "xmin")
            xml_xmin.text = str(xmin)
            xml_ymin = ET.SubElement(xml_bndbox, "ymin")
            xml_ymin.text = str(ymin)
            xml_xmax = ET.SubElement(xml_bndbox, "xmax")
            xml_xmax.text = str(xmax)
            xml_ymax = ET.SubElement(xml_bndbox, "ymax")
            xml_ymax.text = str(ymax)


            # print('\t%s' % info)

            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 3)

            # cv2.imshow('Result', img)
            # cv2.waitKey(500)

    tree = ET.ElementTree(xml_root)
    tree.write(annotation_fldr + '/' + file.split('.')[0] + '.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")

    copyfile(fpath, images_fldr + '/' + file)
