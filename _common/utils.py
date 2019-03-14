import os
import cv2
import numpy as np

def get_impaths_from_path(path):

    image_paths = []

    if os.path.isdir(path):
        for root, subdirs, files in os.walk(path):
            pic_extensions = ('.png', '.PNG', '.jpg', 'JPEG', '.ppm')
            image_paths += [os.path.join(root, file) for file in files if file.endswith(pic_extensions)]
    else:
        image_paths += [path]

    return image_paths

def normalize_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])

    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img


def get_ncs_graph_fpath(config):
    output_dir = 'ncs_graphs'
    output_fpath = os.path.join(output_dir, config['model']['base'] + '.ncsg')

    makedirs_4_file(output_fpath)

    return output_fpath


def get_pb_graph_fpath(config):
    output_dir = 'pb_graphs'
    output_fpath = os.path.join(output_dir, config['model']['base'] + '.pb')

    makedirs_4_file(output_fpath)

    return output_fpath


def get_checkpoint_name(config):
    root = 'chk/{}_{}_{}x{}'.format(config['model']['main_name'], \
                                    config['model']['base'], \
                                    config['model']['infer_shape'][0], \
                                    config['model']['infer_shape'][1])
    name = root + '_ep{epoch:03d}-val_loss{val_loss:.3f}-loss{loss:.3f}' + '.h5'
    return name


def get_mAP_checkpoint_name(config):
    root = 'chk/{}_{}_{}x{}'.format(config['model']['main_name'], \
                                    config['model']['base'], \
                                    config['model']['infer_shape'][0], \
                                    config['model']['infer_shape'][1])
    name = root + '_ep{epoch:03d}-val_loss{val_loss:.3f}-best_mAP{mAP:.3f}' + '.h5'
    return name


DATA_GEN_SRC_VIDEO = 0
DATA_GEN_SRC_IMAGE = 1


def data_generator(input_path):
    video_extensions = ('.mp4', '.webm', '.mov')

    if '/dev/video' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(input_path)

        while True:
            ret_val, image = video_reader.read()

            yield DATA_GEN_SRC_VIDEO, image
        
    elif input_path.endswith(video_extensions):
        video_reader = cv2.VideoCapture(input_path)

        while True:
            ret_val, image = video_reader.read()

            yield DATA_GEN_SRC_IMAGE, image

    else: # do detection on an image or a set of images
        image_paths = get_impaths_from_path(input_path)

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            
            yield DATA_GEN_SRC_IMAGE, image


def makedirs(dirpath):
    if not os.path.isdir(dirpath):
        # Not exist
        os.makedirs(dirpath)


def makedirs_4_file(filepath):
    dirpath = os.path.dirname(filepath)
    makedirs(dirpath)
