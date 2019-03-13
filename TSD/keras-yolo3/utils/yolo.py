import numpy as np
from .utils import preprocess_input, decode_netout, correct_yolo_boxes, do_nms

class YOLO:
    def __init__(self, model, net_h, net_w, anchors, obj_thresh, nms_thresh):
        self.model = model
        self.net_sz = [net_h, net_w]
        self.anchors = anchors
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh

    def infer(self, model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
        # anchors - raw format

        image_h, image_w, _ = images[0].shape
        nb_images = len(images)
        batch_input = np.zeros((nb_images, net_h, net_w, 3))

        anchor_count = len(anchors) // 2

        for i in range(nb_images):
            batch_input[i] = preprocess_input(images[i], net_h, net_w)

        batch_output = model.predict_on_batch(batch_input)
        batch_boxes = [None] * nb_images

        net_output_count = len(batch_output)

        assert net_output_count * 3 == anchor_count, 'Invalid achors count'

        for i in range(nb_images):

            if net_output_count > 1:
                yolos = [batch_output[o][i] for o in range(net_output_count)]
            else:
                yolos = [batch_output[i]]

            boxes = []

            for j in range(len(yolos)):
                l_idx = net_output_count - 1
                r_idx = net_output_count

                yolo_anchors = anchors[(l_idx - j) * 6:(r_idx - j) * 6]

                boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

            correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
            do_nms(boxes, nms_thresh)
            batch_boxes[i] = boxes

        return batch_boxes

