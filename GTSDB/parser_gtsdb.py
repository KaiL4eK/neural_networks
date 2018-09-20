import numpy as np
import cv2
import os
import tqdm
import glob

from lxml import etree as ET

data_root_path = 'data/FullIJCNN2013'
imgs_path = data_root_path
annots_fpath = data_root_path + '/gt.txt'
result_fldr = './annot'

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


for file in tqdm.tqdm(glob.glob(imgs_path + '/*.ppm')):
    
    file = os.path.basename(file)

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
            xml_name.text = str(classId)

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
    tree.write(result_fldr + '/' + file.split('.')[0] + '.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")