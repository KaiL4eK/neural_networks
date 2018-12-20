import mvnc.mvncapi as fx
import time
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from utils import *

import config

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-g', '--graph', help='graph file')
argparser.add_argument('-i', '--input', help='input image path')

DECREASE_SZ = False

args = argparser.parse_args()

def _main_(args):

    input_path = args.input
    graph_fpath = args.graph

    fx.global_set_option(fx.GlobalOption.RW_LOG_LEVEL, 0)

    devices = fx.enumerate_devices()
    if (len(devices) < 1):
        print("Error - no NCS devices detected, verify an NCS device is connected.")
        quit() 

    dev = fx.Device(devices[0])
    
    try:
        dev.open()
    except:
        print("Error - Could not open NCS device.")
        quit()

    print("Hello NCS! Device opened normally.")
    
    with open(graph_fpath, mode='rb') as f:
        graphFileBuff = f.read()

    graph = fx.Graph('graph')

    print("FIFO Allocation")

    if DECREASE_SZ:
        fifoIn, fifoOut = graph.allocate_with_fifos(dev, graphFileBuff, 
                                input_fifo_data_type=fx.FifoDataType.FP16, 
                                output_fifo_data_type=fx.FifoDataType.FP16)
    else:
        fifoIn, fifoOut = graph.allocate_with_fifos(dev, graphFileBuff)

    image_paths = []

    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [os.path.join(input_path, inp_file)]
    else:
        image_paths += [input_path]

    full_time = 0
    image_cnt = 0

    for image_path in tqdm(image_paths):

        img = cv2.imread(image_path)
        inf_img = image_preprocess(img, (config.NETWORK_INPUT_W, config.NETWORK_INPUT_H))

        if DECREASE_SZ:
            inf_img = inf_img.astype(np.float16)
        else:
            inf_img = inf_img.astype(np.float32)

        start = time.time()

        graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, inf_img, None)
        ncs_output, _ = fifoOut.read_elem()
        # print("NCS: ", ncs_output.shape, ncs_output.dtype)

        ncs_output = sigmoid(ncs_output)
        ncs_output = ncs_output.reshape((160, 320, 1))

        if DECREASE_SZ:
            ncs_output = ncs_output.astype(np.float32)

        full_time += time.time() - start
        image_cnt += 1
        
        # print("NCS: ", ncs_output.shape, ncs_output.dtype)

        mask = cv2.resize(ncs_output, (640, 480), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('origin', img)
        cv2.imshow('result', mask)
        if cv2.waitKey() == 27:
            break

    print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, image_cnt / full_time))

    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()

    try:
        dev.close()
    except:
        print("Error - could not close NCS device.")
        quit()

if __name__ == '__main__':
    _main_(args)
