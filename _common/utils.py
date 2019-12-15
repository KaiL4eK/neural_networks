import os
import cv2
import numpy as np
from . import bbox
from scipy.special import expit
from tqdm import tqdm


def get_impaths_from_path(path):

    image_paths = []

    if os.path.isdir(path):
        for root, subdirs, files in os.walk(path):
            pic_extensions = ('.png', '.PNG', '.jpg', 'JPEG', '.ppm')
            image_paths += [os.path.join(root, file)
                            for file in files if file.endswith(pic_extensions)]
    else:
        image_paths += [path]

    return image_paths


def normalize_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img


def image_normalize(image):
    return image / 255


def get_ncs_graph_fpath(config):
    output_dir = 'ncs_graphs'
    output_fpath = os.path.join(output_dir, config['model']['base'] + '.ncsg')

    makedirs_4_file(output_fpath)

    return output_fpath


def get_pb_graph_fpath(config):
    output_dir = '_gen/pb_graphs'
    main_fname = _get_root_name(config)

    output_fpath = os.path.join(output_dir, main_fname)
    makedirs_4_file(output_fpath)

    return output_fpath


def _get_root_name(config):
    if 'tiles' in config['model']:
        root = '{}_{}x{}_t{}'.format(config['model']['base'],
                                    config['model']['infer_shape'][0],
                                    config['model']['infer_shape'][1],
                                    config['model']['tiles']
                                    )
    else:
        root = '{}_{}x{}'.format(config['model']['base'],
                                 config['model']['infer_shape'][0],
                                 config['model']['infer_shape'][1]
                                 )
    return root


def _get_root_checkpoint_name(config):
    return 'chk/{}'.format(_get_root_name(config))


def get_neptune_name(config):
    if 'tiles' in config['model']:
        root = '{}_{}x{}_t{}'.format(config['model']['base'],
                                     config['model']['infer_shape'][0],
                                     config['model']['infer_shape'][1],
                                     config['model']['tiles']
                                     )
    else:
        root = '{}_{}x{}'.format(config['model']['base'],
                                 config['model']['infer_shape'][0],
                                 config['model']['infer_shape'][1]
                                 )
    return root


def get_checkpoint_name(config):
    return _get_root_checkpoint_name(config) + '_ep{epoch:03d}-val_loss{val_loss:.3f}-loss{loss:.3f}' + '.h5'


def get_mAP_checkpoint_name(config):
    return _get_root_checkpoint_name(config) + '_ep{epoch:03d}-val_loss{val_loss:.3f}-best_mAP{mAP:.3f}' + '.h5'

def get_static_checkpoint_name(config):
    return _get_root_checkpoint_name(config) + '.h5'

def get_mAP_checkpoint_static_name(config):
    return _get_root_checkpoint_name(config) + '.h5'


def get_tensorboard_name(config):
    tensorboard_logdir_idx = 0
    root = 'logs/{}_{}x{}_t{}_{}_lr{}_b{}'.format(config['model']['main_name'],
                                                  config['model']['infer_shape'][0],
                                                  config['model']['infer_shape'][1],
                                                  config['model']['tiles'],
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

    if '/dev/video' in input_path:  # do detection on the first webcam
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

    else:  # do detection on an image or a set of images
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
    if not filepath:
        return

    dirpath = os.path.dirname(filepath)
    makedirs(dirpath)


def print_predicted_average_precisions(av_precs):
    for label, average_precision in av_precs.items():
        print(label, '{:.4f}'.format(average_precision))
    mAP = sum(av_precs.values()) / len(av_precs)
    print('mAP: {:.4f}'.format(mAP))

    return mAP


def tiles_get_bbox(img_sz, tile_cnt, tile_idx):
    tile_sz = tiles_get_sz(img_sz, tile_cnt)

    tile_y = 0
    tile_x = tile_idx*tile_sz[1]

    return bbox.BoundBox(tile_x, tile_y, tile_x+tile_sz[1], tile_y+tile_sz[0])


def tiles_get_sz(img_sz, tile_cnt):
    img_h, img_w = img_sz

    if tile_cnt > 2:
        raise NotImplementedError('Tile count > 2 - not supported yet!')

    TILE_REL_SZ = {
        1: (1, 1),
        2: (1, 0.5)
    }

    tile_rel_sz = TILE_REL_SZ[tile_cnt]

    # TODO - Really not dividing by tile count, just hack
    return int(img_h*tile_rel_sz[0]), int(img_w*tile_rel_sz[1])


def tiles_image2batch(image, tile_cnt):
    image_h, image_w, image_c = image.shape
    tile_h, tile_w = tiles_get_sz((image_h, image_w), tile_cnt)

    batch_input = np.zeros((tile_cnt, tile_h, tile_w, image_c))

    for tile_idx in range(tile_cnt):
        tile_bbox = tiles_get_bbox((image_h, image_w), tile_cnt, tile_idx)

        batch_input[tile_idx] = image[tile_bbox.ymin:tile_bbox.ymax,
                                      tile_bbox.xmin:tile_bbox.xmax]

    return batch_input


def get_embedded_img_sz(image_sz, input_hw):
    scale = get_embedded_img_scale(image_sz, input_hw)
    return int(image_sz[0]*scale), int(image_sz[1]*scale)


def get_embedded_img_scale(image_sz, input_hw):
    im_h, im_w = image_sz
    net_h, net_w = input_hw

    return min(float(net_w) / im_w, float(net_h) / im_h)


def image2net_input_sz(image, net_h, net_w):

    new_h, new_w = get_embedded_img_sz(image.shape[0:2], (net_h, net_w))

    resized = cv2.resize(image, (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image.fill(127)

    new_image[(net_h-new_h)//2:(net_h+new_h)//2,
              (net_w-new_w)//2:(net_w+new_w)//2, :] = resized

    return new_image


def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True


def init_session():
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # tf.config.gpu.set_per_process_memory_fraction(rate)
    # tf.config.gpu.set_per_process_memory_growth(True)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_h
        new_w = (image_w*net_h)/image_h

    x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
    y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

    # if len(boxes) == 0:
        # return

    # boxes = np.array(boxes)
    # offsets = np.array([x_offset, y_offset, x_offset, y_offset])
    # scales = np.array([x_scale, y_scale, x_scale, y_scale])
    # sizes = np.array([image_w, image_h, image_w, image_h])
    # boxes[:, :4] = boxes[:, :4]-offsets / scales * sizes
    # boxes[:, :4] = boxes[:, :4].astype(np.int)

    # return boxes
    for i in range(len(boxes)):
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

        boxes[i].xmin, boxes[i].xmax = np.clip([boxes[i].xmin, boxes[i].xmax], 0, image_w)
        boxes[i].ymin, boxes[i].ymax = np.clip([boxes[i].ymin, boxes[i].ymax], 0, image_h)

    return boxes


def do_nms(boxes, nms_thresh):
    bboxes = boxes
    # for box in boxes:
        # bboxes += [bbox.BoundBox(*box)]

    if len(bboxes) > 0:
        nb_class = len(bboxes[0].classes)
    else:
        return bboxes

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in bboxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if bboxes[index_i].classes[c] == 0:
                continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox.bbox_iou(bboxes[index_i], bboxes[index_j]) >= nms_thresh:
                    bboxes[index_j].reset_class_score(c)

    return bboxes


# from numba import njit, prange


def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


# @njit(nopython=True)
def _sigmoid(x):
    # return 1/(1+np.exp(-x))
    return expit(x)


# @njit(parallel=True, nopython=True)
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]

    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4] = _sigmoid(netout[..., 4])
    # obj_exp = np.expand_dims(netout[..., 4], axis=-1)
    obj_exp = netout[..., 4][..., np.newaxis]
    netout[..., 5:] = obj_exp * _sigmoid(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if(objectness <= obj_thresh):
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = np.clip(netout[row, col, b, :4], -1e10, 1e10)
            # x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]

            # boxes += [[x-w/2, y-h/2, x+w/2, y+h/2, classes]]
            boxes += [bbox.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, classes)]
    return boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(
        a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(
        a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

