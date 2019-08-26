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

def get_tensorboard_name(config):
    tensorboard_logdir_idx = 0
    root = 'logs/{}_{}x{}_{}_lr{}_b{}'.format(config['model']['main_name'], \
                                     config['model']['infer_shape'][0], \
                                     config['model']['infer_shape'][1],
                                     config['train']['optimizer'],
                                     config['train']['learning_rate'],
                                     config['train']['batch_size']
    )

    while True:
        name = "%s-%d" % (root, tensorboard_logdir_idx)
        if os.path.exists(name):
            tensorboard_logdir_idx += 1
        else:
            break

    return name


DATA_GEN_SRC_VIDEO = 0
DATA_GEN_SRC_IMAGE = 1


def data_generator(input_path, shuffle=False, cnt_limit=0):
    """
    Parameters
    ----------
    shuffle: Shuffle list of images (not work with video)
    cnt_limit: Maximum number of returned frames
    Returns
    -------
    type: type of input source: DATA_GEN_SRC_VIDEO or DATA_GEN_SRC_IMAGE
    frame: image matrix
    """
    video_extensions = ('.mp4', '.webm', '.mov')

    cntr = 0

    if '/dev/video' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(input_path)

        while True:
            if cnt_limit != 0 and cntr >= cnt_limit:
                break

            ret_val, image = video_reader.read()
            cntr += 1

            yield DATA_GEN_SRC_VIDEO, image
        
    elif input_path.endswith(video_extensions):
        video_reader = cv2.VideoCapture(input_path)

        while True:
            if cnt_limit != 0 and cntr >= cnt_limit:
                break

            ret_val, image = video_reader.read()
            cntr += 1
            
            yield DATA_GEN_SRC_IMAGE, image

    else: # do detection on an image or a set of images
        image_paths = get_impaths_from_path(input_path)

        if shuffle:
            np.random.shuffle(image_paths)

        # the main loop
        for image_path in image_paths:
            if cnt_limit != 0 and cntr >= cnt_limit:
                break

            image = cv2.imread(image_path)
            cntr += 1

            yield DATA_GEN_SRC_IMAGE, image


def makedirs(dirpath):
    if not os.path.isdir(dirpath):
        # Not exist
        os.makedirs(dirpath)


def makedirs_4_file(filepath):
    dirpath = os.path.dirname(filepath)
    makedirs(dirpath)


def print_predicted_average_precisions(av_precs):
    for label, average_precision in av_precs.items():
        print(label, '{:.4f}'.format(average_precision))
    mAP = sum(av_precs.values()) / len(av_precs)
    print('mAP: {:.4f}'.format(mAP))

    return mAP

def get_tiled_image_sz(img_sz, tile_cnt):
    img_h, img_w = img_sz

    if tile_cnt > 2:
        raise NotImplementedError('Tile count > 2 - not supported yet!')

    # TODO - Really not dividing by tile count, just hack
    return img_h, img_w/tile_cnt

def get_embedded_img_sz(image, input_hw):
    new_h, new_w, _ = image.shape
    net_h, net_w = input_hw

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    return new_h, new_w

def image_normalize(image):
    return image / 255

def image2net_input_sz(image, net_h, net_w):

    new_h, new_w = get_embedded_img_sz(image, (net_h, net_w))

    resized = cv2.resize(image, (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image.fill(127)

    new_image[(net_h-new_h)//2:(net_h+new_h)//2,
              (net_w-new_w)//2:(net_w+new_w)//2, :] = resized

    return new_image
