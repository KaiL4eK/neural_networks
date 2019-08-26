import time
import cv2
import argparse
# from _common import ncs
from _common import utils
import json
import yolo
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes
from tensorflow.keras.models import load_model

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-g', '--graph', help='graph path')
argparser.add_argument('-i', '--input', help='input image path')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-f', '--fps', action='store_true', help='FPS estimation mode')

args = argparser.parse_args()


def _main_():
    config_path = args.conf
    graph_fpath = args.graph
    input_path = args.input
    weights_path = args.weights

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    labels = ['sign']
    anchors = config['model']['anchors']

    net_h, net_w = config['model']['infer_shape']
    obj_thresh, nms_thresh = 0.5, 0.45

    if graph_fpath:
        model = ncs.InferNCS(graph_fpath, fp16=False)
    elif weights_path:
        model = load_model(weights_path)
    else:
        _, model, _ = yolo.create_model_new(
            nb_class=len(labels),
            anchors=config['model']['anchors'],
            max_box_per_image=0,
            max_input_size=config['model']['max_input_size'],
            batch_size=config['train']['batch_size'],
            warmup_batches=0,
            ignore_thresh=config['train']['ignore_thresh'],
            grid_scales=config['train']['grid_scales'],
            obj_scale=config['train']['obj_scale'],
            noobj_scale=config['train']['noobj_scale'],
            xywh_scale=config['train']['xywh_scale'],
            class_scale=config['train']['class_scale'],
            base=config['model']['base'],
            base_params=config['model']['base_params'],
            anchors_per_output=config['model']['anchors_per_output'],
            is_freezed=False,
            load_src_weights=False
        )

    data_generator = utils.data_generator(input_path)

    full_time = 0
    processing_cnt = 0
    skip = 0

    for type, image_src in data_generator:

        image = image_src.copy()

        start = time.time()

        # print(image_src.shape)
        # image = cv2.resize(image_src, (0,0), fx=2, fy=2)

        #n_image = utils.normalize_ycrcb(image)
        #boxes = get_yolo_boxes(model, [n_image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]

        boxes = get_yolo_boxes(model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh, no_boxes=args.fps)[0]

        full_time += time.time() - start
        processing_cnt += 1

        if (processing_cnt + 1) % 100 == 0:
            print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, processing_cnt / full_time))

        if skip or type == utils.DATA_GEN_SRC_VIDEO:
            show_delay = 1
        else:
            show_delay = 0
        
        if not args.fps:
            draw_boxes(image, boxes, labels, obj_thresh) 
            draw_boxes(n_image, boxes, labels, obj_thresh) 

            cv2.imshow('1', n_image)
            cv2.imshow('2', image)
            key = cv2.waitKey(show_delay)
            if key == 27:
                break
            elif key == ord(' '):
                skip = 1

    print("Time: %.3f [ms] / FPS: %.1f" % (full_time * 1000, processing_cnt / full_time))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _main_()
