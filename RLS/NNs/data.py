import os
import cv2
import numpy as np
from utils import *

dst_shape_img = (640, 320)


def robofest_data_get_samples():

    rootdir = '../data/robofest_18_lanes'
    png_fpaths = []

    for root, subdirs, files in os.walk(rootdir):
        # print('Root:', root)
        # print('Subdirs:', subdirs)
        # print('Files:', files)

        png_fpaths += [os.path.join(root, file) for file in files if file.endswith('.png')]

    annotation_pairs = [(file, file.replace('.png', '_color_mask.png')) for file in png_fpaths if os.path.exists(file.replace('.png', '_color_mask.png'))]

    orig_imgs = []
    lane_imgs = []

    upper_clip_px = 50

    for src_file, msk_file in annotation_pairs:

        orig_img = cv2.imread(src_file)
        mask_img = cv2.imread(msk_file)

        img_h, img_w, _ = mask_img.shape
        
        orig_img = orig_img[1+upper_clip_px:img_h-1, 1:img_w-1]
        mask_img = mask_img[1+upper_clip_px:img_h-1, 1:img_w-1]

        orig_img = cv2.resize(orig_img, dst_shape_img);
        mask_img = cv2.resize(mask_img, dst_shape_img, interpolation=cv2.INTER_NEAREST);

        layer_colors = set( tuple(v) for m2d in mask_img for v in m2d )
        lane_color = (250, 250, 250)
        
        if lane_color not in layer_colors:
            print('Error')
            exit(1)

        lane_layer = cv2.inRange(mask_img, lane_color, lane_color)

        orig_imgs += [orig_img]
        lane_imgs += [lane_layer]

        # print(src_file)

        # cv2.imshow('src', orig_img)
        # cv2.imshow('mask', lane_layer)
        # cv2.waitKey()


    orig_imgs = np.array(orig_imgs)
    # Extend to 4 dims
    lane_imgs = np.expand_dims(lane_imgs, axis=-1)

    if len(orig_imgs.shape) != 4 or len(lane_imgs.shape) != 4:
        raise Exception('Invalid data shape')

    return (orig_imgs, lane_imgs)

def robofest_data_get_samples_preprocessed(input_shp, output_shp):

    orig_imgs, lane_imgs = robofest_data_get_samples()

    data_count = len(orig_imgs)

    orig_imgs_shp = tuple([data_count] + list(input_shp))
    mask_imgs_shp = tuple([data_count] + list(output_shp))

    inputs  = np.ndarray(orig_imgs_shp, dtype=np.float32)
    outputs = np.ndarray(mask_imgs_shp, dtype=np.float32)

    print(orig_imgs_shp, mask_imgs_shp)

    for i in range(data_count):
        inputs[i]  = image_preprocess(orig_imgs[i], (input_shp[1], input_shp[0]))
        outputs[i] = np.expand_dims(mask_preprocess(lane_imgs[i], (output_shp[1], output_shp[0])), axis=-1)


    return (inputs, outputs)

def test_robofest_data():

    # robofest_data_get_samples_preprocessed()

    orig_imgs, lane_imgs = robofest_data_get_samples()

    for i, lane_im in enumerate(lane_imgs):
        lane_imgs[i] = cv2.resize(lane_im, (orig_imgs.shape[2], orig_imgs.shape[1]))

        print(lane_im.shape)

    print(orig_imgs.shape, lane_imgs.shape)

    for i in range(len(orig_imgs)):

        msk_clr_image = np.zeros_like(orig_imgs[i], np.uint8)
        msk_clr_image[:] = (255, 0, 0)

        img1_bg = cv2.bitwise_and(orig_imgs[i], orig_imgs[i], mask=cv2.bitwise_not(lane_imgs[i]))
        img2_fg = cv2.bitwise_and(msk_clr_image, msk_clr_image, mask=lane_imgs[i])

        full_img = cv2.add(img1_bg,img2_fg)

        cv2.imshow('1', full_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    test_robofest_data()

