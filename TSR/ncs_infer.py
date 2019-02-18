import core
import time
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import os
from _common import ncs
from _common import utils
import data
import json

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-g', '--graph', help='graph path')
argparser.add_argument('-i', '--input', help='input image path')

args = argparser.parse_args()


def _main_():
    config_path = args.conf
    graph_fpath = args.graph
    input_path = args.input

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    classes = data.get_classes(config['train']['cache_name'])

    ncs_dev = ncs.InferNCS(graph_fpath, fp16=False)

    image_paths = utils.get_impaths_from_path(input_path)

    full_time = 0
    image_cnt = 0

    for image_path in tqdm(image_paths):

        # print('Processing: {}'.format(image_path))
        image_src = cv2.imread(image_path)

        net_input_shape = (config['model']['input_side_sz'],
                           config['model']['input_side_sz'])

        start = time.time()

        image = cv2.resize(image_src, net_input_shape)
        image = data.normalize(image)

        ncs_output = ncs_dev.infer(image)
        ncs_output = ncs_output.astype(np.float32)

        full_time += time.time() - start
        image_cnt += 1

        ncs_output.reshape((len(classes)))
        max_idx = np.argmax(ncs_output)

        # print("NCS: ", ncs_output.shape, ncs_output.dtype)
        print('Result: {} / {}'.format(classes[max_idx], max_idx))

        show_delay = 0

        cv2.imshow('res', image_src)
        if cv2.waitKey(show_delay) == 27:
            break

    print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, image_cnt / full_time))


if __name__ == '__main__':
    _main_()
