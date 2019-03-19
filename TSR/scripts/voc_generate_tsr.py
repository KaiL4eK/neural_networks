import sys
sys.path.append('..')

from _common.utils import makedirs
from _common.voc import parse_voc_annotation, split_by_objects

import os
import numpy as np
import cv2

import argparse
argparser = argparse.ArgumentParser(description='Help =)')
argparser.add_argument('-i', '--input', help='path to folder to flip')
argparser.add_argument('-a', '--annot', help='path to folder to flip')
argparser.add_argument('-o', '--output', help='path to folder to output')

args = argparser.parse_args()

def main():
    images_path = args.input
    annot_path = args.annot
    output_path = args.output

    ints, labels = parse_voc_annotation([annot_path], [images_path], None)

    stats = {}

    makedirs(output_path)

    for instance in ints:
        img = cv2.imread(instance['filename'])

        for obj in instance['object']:
            class_name = obj['name']
            class_dir = os.path.join(output_path, class_name)


            if class_name not in stats:
                if os.path.isdir(class_dir):
                    files = os.listdir(class_dir)

                    max_idx = 0
                    for file in files:
                        file_idx = int(file.split('.')[0])
                        max_idx = np.max([max_idx, file_idx])

                    stats[class_name] = max_idx + 1
                else:
                    makedirs(class_dir)
                    stats[class_name] = 1
            else:
                stats[class_name] += 1

            img_part = img[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]

            output_fpath = os.path.join(class_dir, '{}.png'.format(stats[class_name]))

            cv2.imwrite(output_fpath, img_part)



if __name__ == '__main__':
    main()
