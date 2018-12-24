import time
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import os
import ncs

from utils import *
import config

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-i', '--input', help='input image path')

args = argparser.parse_args()

def _main_(args):

    input_path = args.input

    ncs_dev = ncs.InferNCS(config.NCS_GRAPH)

    image_paths = []

    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [os.path.join(input_path, inp_file)]
    else:
        image_paths += [input_path]

    full_time = 0
    image_cnt = 0

    for image_path in tqdm(image_paths):

        print('Processing: {}'.format(image_path))
        img = cv2.imread(image_path)

        start = time.time()
        input_sz = (config.NETWORK_INPUT_W, config.NETWORK_INPUT_H)

        inf_img = image_preprocess(img.copy(), input_sz)

        ncs_output = ncs_dev.infer(inf_img)
        ncs_output = ncs_output.astype(np.float32)

        full_time += time.time() - start
        image_cnt += 1
        
        # print("NCS: ", ncs_output.shape, ncs_output.dtype)

        mask = cv2.resize(ncs_output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow('res', np.hstack([img, mask]))
        if cv2.waitKey() == 27:
            break

    print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, image_cnt / full_time))


if __name__ == '__main__':
    _main_(args)
