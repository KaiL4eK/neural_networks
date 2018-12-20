import sys
sys.path.insert(0, 'TensorRT/src/')
import tensorNet
import argparse
import os
from tqdm import tqdm
import cv2
import time
import numpy as np

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-i', '--input', help='path to an image(-s)', default=None)
args = argparser.parse_args()

import config
from utils import image_preprocess

def _main_():

    input_path = args.input

    uff_fpath = 'TensorRT/uff/{}.uff'.format(config.NET_BASENAME)
    engine_fpath = 'TensorRT/engines/{}.trt'.format(config.NET_BASENAME)

    if os.path.exists(engine_fpath):
        engine = tensorNet.createTrtFromPlan(engine_fpath)
    else:
        if os.path.exists(uff_fpath):
            engine = tensorNet.createTrtFromUFF(uff_fpath, config.INPUT_TENSOR_NAMES[0], 'activation_1/Sigmoid')
            tensorNet.saveEngine(engine, engine_fpath)
        else:
            print('No .uff file!')
            exit(1)

    image_paths = []
    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [os.path.join(input_path, inp_file)]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG', '.ppm'])]

    processing_count = 0
    sum_time = 0

    network_input_shp = (config.NETWORK_INPUT_W, config.NETWORK_INPUT_H, config.NETWORK_INPUT_C)

    render_mode = True

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)

        start_time = time.time()

        image_h, image_w, _ = image.shape

        input_img = image_preprocess(image, network_input_shp)

        # Convert 2 CHW
        image_chw = np.moveaxis(input_img, -1, 0)
        image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)

        tensorNet.inference(engine, image_chw)

        mask_result = np.zeros((160, 320, 1), dtype=np.float32)
        print(mask_result.shape)
        tensorNet.getOutput(engine, 0, mask_result)

        image_chw = np.moveaxis(input_img, -1, 0)

        sum_time += time.time() - start_time
        processing_count += 1

        if render_mode:
            cv2.imshow('result', np.uint8(mask_result))
            cv2.imshow('input', image)
            
            if cv2.waitKey(0) == 27:
                break  # esc to quit

    fps = processing_count / sum_time
    print('Result: {}'.format(fps))

if __name__ == '__main__':
    _main_()
