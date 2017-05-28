from __future__ import print_function

import matplotlib.pyplot as plt 
import matplotlib.animation as animation

import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

from net import *
import argparse
import moviepy.editor as mpy

# from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('filepath', action='store', help='Path to video file to process')
parser.add_argument('-f', '--fps', action='store_true', help='Check fps')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')
parser.add_argument('-n', '--negative', action='store_true', help='Negative creation')
parser.add_argument('-g', '--gif', action='store_true', help='Create gif')

args = parser.parse_args()

data_path = 'raw/'

try:
	xrange
except NameError:
	xrange = range

def build_gif(imgs, show_gif=True, save_gif=True, title=''):
	print('Creating GIF file')

	clip = mpy.ImageSequenceClip(imgs, fps=20)
	clip.write_gif('anim.gif', fps=20)

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.set_axis_off()
 
	# ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)
 
	# im_ani = animation.ArtistAnimation(fig, ims, interval=800, repeat_delay=0, blit=False)
 
	# if save_gif:
	# 	im_ani.save('animation.gif', writer='imagemagick')
 
	# if show_gif:
	# 	plt.show()
 

def process_naming(frame, model):
	img_class = model.predict(np.array([preprocess_img(frame)]))

	img_class = img_class[0]
	# print(img_class)

	class_index = np.argmax(img_class)
	class_value = img_class[class_index]

	if class_value > 0.95 and class_index != 0:
		# print(class_value)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, class_list[class_index], (10,70), font, 2, (255,0,0), 3)

def get_next_negative_id():
	maximum_id = 0

	data_path = '../negative'
	images = os.listdir(data_path)

	for imagefile in images:
		info   = imagefile.split(';')
		maximum_id = max(maximum_id, int(info[0]))

	return maximum_id+1

def execute_model():
	model = get_network_model()
	gif_imgs = []

	if args.pic:
		
		frame = cv2.imread(args.filepath)
		if frame is None:
			print('Failed to open file')
			exit(1)
		
		model.load_weights(args.weights)

		process_naming(frame, model)

		cv2.imshow('frame',frame)
		cv2.waitKey(0)
		exit(1)
	else:

		cap = cv2.VideoCapture(args.filepath)
		if cap is None or not cap.isOpened():
			print('Failed to open file')
			exit(1)

		ret, frame = cap.read()
		if frame is None:
			print('Failed to read frame')
			exit(1)

		model.load_weights(args.weights)

		if args.fps:
			bbox_obtain_time = 0
			num_frames = 120
			start = time.time()
			for i in xrange(0, num_frames) :
				ret, frame = cap.read()
				if frame is None:
					exit(1)

				start_bbox = time.time()

				process_naming(frame, model)
				
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			seconds = time.time() - start
			print('Mean time per frame : {0} ms'.format(seconds / num_frames * 1000))
			print('Estimated frames per second : {0}'.format(num_frames / seconds))
			print('Bbox obtain mean time : {0} ms'.format(bbox_obtain_time / num_frames * 1000))

		elif args.negative:	
			negative_id = get_next_negative_id()
			while(cap.isOpened()):
				ret, frame = cap.read()
				if frame is None:
					exit(1)

				frame_draw = np.copy(frame)
				process_naming(frame_draw, model)
				
				cv2.imshow('frame',frame_draw)

				wait_res = cv2.waitKey(0)
				if wait_res & 0xFF == ord(' '):
					filename = '../negative/{};0;0;0;0;neg.png'.format(negative_id)
					print('Saving to file: {}'.format(filename))
					negative_id += 1

					cv2.imwrite(filename, frame)

				if wait_res & 0xFF == ord('q'):
					break
		else:
			while(cap.isOpened()):
				ret, frame = cap.read()
				if frame is None:
					exit(1)

				process_naming(frame, model)
				
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

				frame = cv2.resize(frame, (80, 80), interpolation = cv2.INTER_LINEAR)

				if args.gif:
					gif_imgs.append(frame)

		cap.release()
		cv2.destroyAllWindows()

	if args.gif:
		build_gif(gif_imgs)

if __name__ == '__main__':
	execute_model()
