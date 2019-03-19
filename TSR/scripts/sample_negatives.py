import os
import cv2
import random

import argparse
argparser = argparse.ArgumentParser(description='Help =)')
argparser.add_argument('-i', '--input', help='path to folder to flip')
argparser.add_argument('-o', '--output', help='path to folder to output')
argparser.add_argument('-c', '--count', help='count of sampled parts')

args = argparser.parse_args()

part_sz = (96, 96)
out_part_sz = (32, 32)


def main():
    input_fldr = args.input
    output_fldr = args.output

    if not os.path.exists(input_fldr):
        print('Input folder is incorrect')

    if not os.path.exists(output_fldr):
        os.mkdir(output_fldr)

    img_fpaths = [os.path.join(input_fldr, in_fname) for in_fname in os.listdir(input_fldr)]

    idx = 0

    for in_fpath in img_fpaths:
        img = cv2.imread(in_fpath)
        img_h, img_w, _ = img.shape

        x1 = random.randint(0, img_w - part_sz[0]-1)
        y1 = random.randint(0, img_h - part_sz[1]-1)

        x2, y2 = x1 + part_sz[0] - 1, y1 + part_sz[1] - 1

        img_part = img[y1:y2,x1:x2]

        output_fpath = os.path.join(output_fldr, '{}.png'.format(idx))
        idx += 1

        cv2.imwrite(output_fpath, img_part)

        # cv2.imshow('1', img)
        # cv2.imshow('2', img_part)
        # cv2.waitKey(0)

if __name__ == '__main__':
    main()
    