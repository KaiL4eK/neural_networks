import cv2

import argparse

parser = argparse.ArgumentParser(description='Process video by hands to get nagatives')
parser.add_argument('filepath', action='store', help='Path to video file to process')
parser.add_argument('id', action='store', help='ID to start from')

args = parser.parse_args()

try:
    xrange
except NameError:
    xrange = range

def main():
	cap = cv2.VideoCapture(args.filepath)
	negative_id = int(args.id)

	while(cap.isOpened()):

		ret, frame = cap.read()
		if frame is None:
			exit(1)

		cv2.imshow('frame',frame)
		wait_res = cv2.waitKey(0)
		if wait_res & 0xFF == ord('q'):
			break
		
		if wait_res & 0xFF == ord(' '):
			filename = 'negative/{};0;0;0;0;neg.png'.format(negative_id)
			print('Saving to file: {}'.format(filename))
			negative_id += 1

			cv2.imwrite(filename, frame)


if __name__ == '__main__':
	main()

