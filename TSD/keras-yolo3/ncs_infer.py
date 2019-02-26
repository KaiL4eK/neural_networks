import core
import time
import cv2
import argparse
from _common import ncs
from _common import utils
import json
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes

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
        config = json.load(config_buffer)

    config['model']['labels'] = ['brick', 'forward', 'forward and left', 'forward and right', 'left', 'right']

    net_h, net_w = config['model']['infer_shape']
    obj_thresh, nms_thresh = 0.8, 0.45

    ncs_model = ncs.InferNCS(graph_fpath, fp16=False)

    data_generator = utils.data_generator(input_path)

    show_delay = 1000

    full_time = 0
    processing_cnt = 0

    for image_src in data_generator:

        start = time.time()

        print(image_src.shape)
        # image = cv2.resize(image_src, (0,0), fx=2, fy=2)

        boxes = get_yolo_boxes(ncs_model, [image_src], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        full_time += time.time() - start
        processing_cnt += 1

        image = image_src
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 

        cv2.imshow('1', image)
        if cv2.waitKey(show_delay) == 27:
            break

    print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, processing_cnt / full_time))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _main_()
