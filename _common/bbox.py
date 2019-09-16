import numpy as np
import cv2
from .colors import get_color


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, classes = None, label_name = None, label_idx = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

        self.classes = classes
        self.class_name = label_name
        self.class_idx = label_idx

        # Means score of best class
        if self.classes:
            self.label = np.argmax(self.classes)
            self.score = self.classes[self.label]
        else:
            self.label = -1
            self.score = -1

    def get_best_class_score(self):
        return self.score

    def reset_class_score(self, class_idx):
        if self.classes:
            self.classes[class_idx] = 0
            self.label = np.argmax(self.classes)
            self.score = self.classes[self.label]
    
    def __str__(self):
        return "[{}:{}, {}:{} / prob: {} / classes: {}]".format(self.ymin, self.ymax, self.xmin, self.xmax, self.classes)
        
    def __repr__(self):
        return "[{}:{}, {}:{} / prob: {} / classes: {}]".format(self.ymin, self.ymax, self.xmin, self.xmax, self.classes)
    
    def get_str(self):
        return "[{}:{}, {}:{} / prob: {} / classes: {}]".format(self.ymin, self.ymax, self.xmin, self.xmax, self.classes)

    def intersect(self, bbox):
        x = max(self.xmin, bbox.xmin)
        y = max(self.ymin, bbox.ymin)
        w = min(self.xmax, bbox.xmax) - x
        h = min(self.ymax, bbox.ymax) - y
        if w < 0 or h < 0: return None
        return BoundBox(x, y, x+w, y+h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    try:
        result = float(intersect) / union
    except:
        result = 0

    return result


def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_best_class_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        if label >= 0:
            font_scl = 1.5e-3 * image.shape[0]
            thickness = 2

            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scl, thickness)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin],
                               [box.xmin-3,        box.ymin-height-26],
                               [box.xmin+width+13, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=3)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin+13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scl,
                        color=(0,0,0),
                        thickness=thickness)

    return image

def draw_ros_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = box.Class
        label = 0

        # for i in range(len(labels)):
        #     if box.classes[i] > obj_thresh:
        #         if label_str != '': label_str += ', '
        #         score = box.classes[np.argmax(box.classes)]
        #         label_str += (labels[i] + ' ' + str(round(score*100, 2)) + '%')
        #         label = i
        #     if not quiet: print(label_str)

        if label >= 0:
            font_scl = 1.5e-3 * image.shape[0]
            thickness = 2

            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scl, thickness)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin],
                               [box.xmin-3,        box.ymin-height-26],
                               [box.xmin+width+13, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin+13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scl,
                        color=(0,0,0),
                        thickness=thickness)

    return image
