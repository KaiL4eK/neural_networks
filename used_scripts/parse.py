from __future__ import print_function

import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('filepath', action='store', help='Path to video file to process')

args = parser.parse_args()

def parse_file():


	img_ref = cv2.imread(args.filepath)

	pic_path = os.path.splitext(args.filepath)[0]+".jpg"
	print(pic_path)

	img = cv2.imread(pic_path)

	cv2.imshow('1', img)
	cv2.imshow('2', img_ref)

	cv2.waitKey(0)

if __name__ == '__main__':
	parse_file()

