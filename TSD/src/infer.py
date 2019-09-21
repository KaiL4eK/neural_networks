import time
import cv2
import argparse
from _common import utils
import json
import yolo
from _common.bbox import draw_boxes
from tensorflow.keras.models import load_model

argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-i', '--input', help='input image path')
argparser.add_argument('-w', '--weights', help='weights path')
argparser.add_argument('-f', '--fps', action='store_true', help='FPS estimation mode')

args = argparser.parse_args()


def _main_():
    config_path = args.conf
    input_path = args.input
    weights_path = args.weights

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    utils.init_session(0.3)

    labels = ['sign']

    net_h, net_w = config['model']['infer_shape']
    obj_thresh, nms_thresh = 0.5, 0.45

    config['model']['labels'] = labels
    yolo_model = yolo.YOLO_Model(
        config['model']
    )

    if weights_path:
        yolo_model.load_weights(weights_path)

    data_generator = utils.data_generator(input_path)

    full_time = 0
    processing_cnt = 0
    skip = 0
    
    fps_stats = {}

    for type, image_src in data_generator:

        image = image_src.copy()

        start = time.time()
        
        if args.fps:
            boxes, times = yolo_model.timed_infer_image(image)
            for k, v in times.items():
                if k in fps_stats:
                    fps_stats[k] += v
                else:
                    fps_stats[k] = v
                
                print('Op: {} / Time: {} ms'.format(k, fps_stats[k]/(processing_cnt+1)))
            print('===============================')
        else:
            boxes = yolo_model.infer_image(image)

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

            # cv2.imshow('1', n_image)
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
