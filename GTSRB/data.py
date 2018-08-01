import os
import numpy as np
import cv2

data_root  = 'data/Final_Training/Images'
import readTrafficSigns as ts

save_data_path = '.'

npy_imgs_filename = 'imgs_train.npy'
npy_lbls_filename = 'lbls_train.npy'

npy_img_height = 48
npy_img_width = 48

sum_0 = 0
sum_1 = 0

def create_train_data():
	images, labels = ts.readTrafficSigns(data_root)

	total = len(images)

	npy_img_list = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
	npy_lbl_list = np.ndarray((total),         							dtype=np.uint8)	

	for i, img in enumerate(images):
		# (h, w, c)
		# print(img.shape)

		c_img = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)

		# Conversion just for OpenCV rendering
		# c_img = cv2.cvtColor(c_img, cv2.COLOR_RGB2BGR)
		# cv2.imshow('res', c_img)
		# cv2.waitKey(1000)

		npy_img_list[i] = c_img
		npy_lbl_list[i] = labels[i]

		# sum_0 += img.shape[0]
		# sum_1 += img.shape[1]

	# print( sum_0 / len(images), sum_1 / len(images) )


	np.save(os.path.join(save_data_path, npy_imgs_filename), npy_img_list)
	np.save(os.path.join(save_data_path, npy_lbls_filename), npy_lbl_list)

	print('Saving to .npy {} files done'.format(total))


def npy_data_load_images():
    return np.load(os.path.join(save_data_path, npy_imgs_filename))


def npy_data_load_labels():
    return np.load(os.path.join(save_data_path, npy_lbls_filename))


if __name__ == '__main__':
    create_train_data()
