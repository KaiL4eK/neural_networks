import rosbag
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
argparser.add_argument('-b', '--bag', help='bag file')
args = argparser.parse_args()

def _main_():

    ncs_dev = ncs.InferNCS(config.NCS_GRAPH)

    bag = rosbag.Bag(args.bag)

    print(bag)
    
    full_time = 0
    image_cnt = 0

    video_writer = cv2.VideoWriter('result.mp4',
                                   cv2.VideoWriter_fourcc(*'FMP4'), 
                                   25.0, 
                                   (320 * 2, 160))

    for topic, msg, t in bag.read_messages(topics=['/nuc/image_raw/compressed']):

        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        start = time.time()
        input_sz = (config.NETWORK_INPUT_W, config.NETWORK_INPUT_H)

        inf_img = image_preprocess(image_np, input_sz)

        ncs_output = ncs_dev.infer(inf_img)
        ncs_output = ncs_output.astype(np.float32)

        full_time += time.time() - start
        image_cnt += 1

        # mask = cv2.resize(ncs_output, input_sz, interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(ncs_output, cv2.COLOR_GRAY2BGR)
        
        # print(mask.shape)
        # print(inf_img.shape)
        
        two_frames = np.hstack([inf_img, mask])

        video_writer.write((two_frames * 255).astype(np.uint8))

        cv2.imshow('res', two_frames)
        if cv2.waitKey(1) == 27:
            break

    video_writer.release()

    print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, image_cnt / full_time))
    bag.close()


if __name__ == '__main__':
    _main_()
