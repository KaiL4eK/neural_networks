import numpy as np
from .utils import preprocess_input, decode_netout, correct_yolo_boxes, do_nms

class YOLO:
    def __init__(self, model, net_h, net_w, anchors, obj_thresh, nms_thresh):
        self.model = model
        self.net_sz = [net_h, net_w]
        self.anchors = anchors
        self.anchor_count = len(self.anchors) // 2

        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh

    def make_infer(self, images):
        # anchors - raw format

        image_h, image_w, _ = images[0].shape
        nb_images = len(images)
        batch_input = np.zeros((nb_images, self.net_sz[0], self.net_sz[1], 3))

        for i in range(nb_images):
            batch_input[i] = preprocess_input(images[i], self.net_sz[0], self.net_sz[1])

        batch_output = self.model.predict_on_batch(batch_input)
        batch_boxes = [None] * nb_images

        net_output_count = len(batch_output)

        for i in range(nb_images):

            if net_output_count > 1:
                yolos = [batch_output[o][i] for o in range(net_output_count)]
            else:
                yolos = [batch_output[i]]

            boxes = []

            for j in range(len(yolos)):
                l_idx = net_output_count - 1
                r_idx = net_output_count

                yolo_anchors = self.anchors[(l_idx - j) * 6:(r_idx - j) * 6]

                boxes += decode_netout(yolos[j], yolo_anchors, self.obj_thresh, self.net_sz[0], self.net_sz[1])

            correct_yolo_boxes(boxes, image_h, image_w, self.net_sz[0], self.net_sz[1])

            # for yolo_bbox in boxes:
                # print("Before NMS: {}",format(yolo_bbox.get_str()))

            do_nms(boxes, self.nms_thresh)

            # for yolo_bbox in boxes:
                # print("After NMS: {}",format(yolo_bbox.get_str()))

            batch_boxes[i] = boxes

        return batch_boxes
