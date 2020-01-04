from functools import wraps, reduce
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D
from tensorflow.keras.models import Model
from tqdm import tqdm

from _common import utils
from _common import backend

import os


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid_hw,
                 batch_size, warmup_batches, iou_thresh,
                 grid_scale, obj_scale, noobj_scale,
                 xywh_scale, class_scale, debug=False,
                 **kwargs):
        # make the model settings persistent
        self.iou_thresh = iou_thresh
        self.warmup_batches = warmup_batches
        self.anchors_per_output = len(anchors) // 2
        self.anchors = tf.constant(
            anchors, dtype='float', shape=[1, 1, 1, self.anchors_per_output, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale

        print('YoloLayer anchors: {} / max_grid: {}'.format(anchors, max_grid_hw))

        self.debug = debug
        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid_hw

        cell_x = tf.cast(
            tf.reshape(
                tf.tile(
                    tf.range(max_grid_w),
                    [max_grid_h]
                ),
                (1, max_grid_h, max_grid_w, 1, 1)
            ), tf.float32
        )

        cell_y = tf.cast(
            tf.reshape(
                tf.tile(
                    tf.range(max_grid_h),
                    [max_grid_w]
                ),
                (1, max_grid_w, max_grid_h, 1, 1)
            ), tf.float32
        )
        cell_y = tf.transpose(cell_y, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1),
            [batch_size, 1, 1, self.anchors_per_output, 1]
        )

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(YoloLayer, self).build(input_shape)

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        t_batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)

        # adjust the shape of the y_predict [batch, grid_h, grid_w, n_anchors, 4+1+nb_class]
        y_pred = tf.reshape(
            y_pred, tf.concat(
                [tf.shape(y_pred)[:3], tf.constant([self.anchors_per_output, -1])],
                axis=0
            )
        )

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        # batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(
            tf.cast([grid_w, grid_h], tf.float32),
            [1, 1, 1, 1, 2]
        )

        # grid_factor = tf.Print(grid_factor, [grid_h, grid_w, tf.shape(y_pred)[:3]],
        #                         message='sz \t', summarize=1000)

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(
            tf.cast([net_w, net_h], tf.float32),
            [1, 1, 1, 1, 2]
        )

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] +
                       tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        # t_wh
        pred_box_wh = y_pred[..., 2:4]
        # adjust confidence
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)
        # adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        # true_box_class = tf.argmax(y_true[..., 5:], -1)
        true_box_class = y_true[..., 5:]

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(
            tf.exp(pred_box_wh) * self.anchors / net_factor, 4
        )

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_delta *= tf.expand_dims(
            tf.cast(best_ious < self.iou_thresh, tf.float32),
            4
        )

        """
        Compute some online statistics
        """
        # true_xy = true_box_xy / grid_factor
        # true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        # true_wh_half = true_wh / 2.
        # true_mins = true_xy - true_wh_half
        # true_maxes = true_xy + true_wh_half

        # pred_xy = pred_box_xy / grid_factor
        # pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

        # pred_wh_half = pred_wh / 2.
        # pred_mins = pred_xy - pred_wh_half
        # pred_maxes = pred_xy + pred_wh_half

        # intersect_mins = tf.maximum(pred_mins,  true_mins)
        # intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        # intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        # intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        # true_areas = true_wh[..., 0] * true_wh[..., 1]
        # pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        # union_areas = pred_areas + true_areas - intersect_areas
        # iou_scores = tf.truediv(intersect_areas, union_areas)
        # iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        # count = tf.reduce_sum(object_mask)
        # count_noobj = tf.reduce_sum(1 - object_mask)
        # detect_mask = tf.cast((pred_box_conf*object_mask) >= 0.5, tf.float32)
        # class_mask = tf.expand_dims(
        #     tf.cast(
        #         tf.equal(
        #             tf.argmax(pred_box_class, -1),
        #             true_box_class
        #         ), tf.float32
        #     ),
        #     4
        # )

        # recall50 = tf.reduce_sum(
        #     tf.cast(iou_scores >= 0.5, tf.float32) *
        #     detect_mask * class_mask
        # ) / (count + 1e-3)

        # recall75 = tf.reduce_sum(
        #     tf.cast(iou_scores >= 0.75, tf.float32) *
        #     detect_mask * class_mask
        # ) / (count + 1e-3)

        # avg_iou = tf.reduce_sum(
        #     iou_scores
        # ) / (count + 1e-3)

        # avg_obj = tf.reduce_sum(
        #     pred_box_conf * object_mask
        # ) / (count + 1e-3)

        # avg_noobj = tf.reduce_sum(
        #     pred_box_conf * (1-object_mask)
        # ) / (count_noobj + 1e-3)

        # avg_cat = tf.reduce_sum(
        #     object_mask * class_mask
        # ) / (count + 1e-3)

        """
        Warm-up training
        """
        # batch_seen = tf.assign_add(batch_seen, 1.)

        # true_box_xy, true_box_wh, xywh_mask = tf.cond(
        #     tf.less(batch_seen, self.warmup_batches+1),
        #     lambda: [true_box_xy +
        #              (0.5 +
        #               self.cell_grid[:, :grid_h, :grid_w, :, :]
        #               ) * (1-object_mask),
        #              true_box_wh +
        #              tf.zeros_like(true_box_wh) *
        #              (1-object_mask),
        #              tf.ones_like(object_mask)],
        #     lambda: [true_box_xy,
        #              true_box_wh,
        #              object_mask]
        # )
        # xywh_mask = object_mask

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) * self.xywh_scale

        xy_delta = object_mask * wh_scale * (pred_box_xy-true_box_xy)
        wh_delta = object_mask * wh_scale * (pred_box_wh-true_box_wh)
        conf_delta = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_box_conf, logits=pred_box_conf) * self.obj_scale + \
                    (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class) * self.class_scale

        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))
        loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))

        # print('\n')
        # print('Loss XY: {}'.format(loss_xy))
        # print('Loss WH: {}'.format(loss_wh))
        # print('Loss Conf: {}'.format(loss_conf))
        # print('Loss Class: {}'.format(loss_class))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        # if self.debug:
        #     loss = tf.Print(loss, [grid_h, avg_obj],
        #                     message='avg_obj \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, avg_noobj],
        #                     message='avg_noobj \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, avg_iou],
        #                     message='avg_iou \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, avg_cat],
        #                     message='avg_cat \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, recall50],
        #                     message='recall50 \t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, recall75],
        #                     message='recall75 \t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, count],
        #                     message='count \t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
        #                            tf.reduce_sum(loss_wh),
        #                            tf.reduce_sum(loss_conf),
        #                            tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss * self.grid_scale / t_batch_size

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


class YOLO_Model:
    def __init__(
        self,
        model_config: dict,
        train_config=None
    ):
        self.model_config = model_config
        self.train_config = train_config

        self.labels = model_config['labels']
        self.anchors = model_config['anchors']
        self.downsample = model_config['downsample']
        self.anchors_per_output = int(len(self.anchors) / 2 / len(self.downsample))
        # self.anchors_per_output = model_config['']

        self.nms_thresh = 0.5
        self.obj_thresh = 0.5

        self.nb_classes = int(len(self.labels))
        self.infer_sz = model_config['infer_shape']

        backends = {
                    'Tiny3':       (backend.Tiny_YOLOv3,       "yolov3-tiny.h5"),
                    'SmallTiny3':  (backend.Small_Tiny_YOLOv3, ""),
                    'Darknet53':   (backend.Darknet53,         "yolov3_exp.h5"),
                    'Darknet19':   (backend.Darknet19,         "yolov2.h5"),
                    'MbN2':        (backend.MobileNetV2,       ""),
                    'SmallMbN2':   (backend.SmallMobileNetV2,  ""),
                    'RF_MbN2':     (backend.RFMobileNetV2,     ""),
                    'RF2_MbN2':    (backend.RF2MobileNetV2,    ""),
                    'MadNet1':     (backend.MadNetv1,          ""),
                    'SqueezeNet':  (backend.SqueezeNet,        ""),
                    'Xception':    (backend.Xception,          "")
                    }

        base = model_config['base']

        if base not in backends:
            print('No such base: {}'.format(base))
            return None

        print('Loading "{}" model'.format(base))

        # If we gonna train - set input (None, None, 3)
        train_shape = (None, None, 3) if self.train_config else (*self.infer_sz, 3)

        backend_options = {
            'train_shape':  train_shape,
            'base_params':  model_config['base_params']
        }

        self.backend = backends[base][0](**backend_options)

        if train_config:
            weights_path = os.path.join('src_weights', backends[base][1])
            if backends[base][1] and os.path.exists(weights_path) and train_config['load_src_weights']:
                print('Loading {}'.format(backends[base][1]))
                self.backend.model.load_weights(
                    weights_path,
                    by_name=True
                )

        # Create infer model
        self.pred_layers = []
        for idx, out in enumerate(self.backend.outputs):
            pred_yolo = Conv2D(filters=self.anchors_per_output*(4+1+self.nb_classes),
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            name='PredictionLayer_'+str(idx),
                            kernel_initializer='lecun_normal'
                            )(out)
            self.pred_layers += [pred_yolo]

        self.infer_model = Model(self.backend.inputs, self.pred_layers)

        if self.train_config:
            self._setup_trainer()

    def _setup_trainer(self):
        apo = self.anchors_per_output

        max_box_per_image = self.train_config['mbpi']

        true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
        true_yolos = []
        for i in range(len(self.backend.outputs)):
            # grid_h, grid_w, nb_anchor, 5+nb_class
            true_yolo = Input(shape=(None, None, apo, 4+1+self.nb_classes),
                            name='input_true_yolo_{}_x{}'.format(i, self.backend.downgrades[i]))
            true_yolos += [true_yolo]

        yolo_anchors = []
        for i in reversed(range(len(self.backend.outputs))):
            anchors_start_idx = i*2*apo
            anchors_end_idx = (i+1)*2*apo
            yolo_anchors += [self.anchors[anchors_start_idx:anchors_end_idx]]

        image_input = self.backend.inputs[0]
        loss_yolos = []
        for idx, out in enumerate(self.backend.outputs):
            max_grid_hw = np.array([self.train_config['max_input_size'][0] // self.backend.downgrades[idx],
                                    self.train_config['max_input_size'][1] // self.backend.downgrades[idx]])

            loss_yolo = YoloLayer(yolo_anchors[idx],
                                  max_grid_hw,
                                  self.train_config['batch_size'],
                                  0,
                                  self.train_config['iou_thresh'],
                                  self.train_config['grid_scales'][idx],
                                  self.train_config['obj_scale'],
                                  self.train_config['noobj_scale'],
                                  self.train_config['xywh_scale'],
                                  self.train_config['class_scale'])([image_input, self.pred_layers[idx], true_yolos[idx], true_boxes])

            loss_yolos += [loss_yolo]

        self.train_model = Model([image_input, true_boxes] + true_yolos, loss_yolos)

        # import tensorflow.keras.backend as K
        # print('>>>> {}'.format(K.shape(out)[1]))

    def load_weights(self, weights_fpath: str):
        if os.path.exists(weights_fpath):
            print("\nLoading weights {}".format(weights_fpath))
            self.infer_model.load_weights(weights_fpath, by_name=True)
        else:
            print('\nInvalid weights path" {}'.format(weights_fpath))

    def infer_image(self, image):
        tile_count = self.model_config.get('tiles', 1)

        if tile_count == 1:
            return self._infer_on_batch(np.expand_dims(image, axis=0))[0]
        else:
            images_batch = utils.tiles_image2batch(image, tile_count)
            pred_batch_boxes = self._infer_on_batch(images_batch)

            corrected_boxes = []
            for tile_idx, pred_boxes in enumerate(pred_batch_boxes):
                tile_bbox = utils.tiles_get_bbox(image.shape[0:2], tile_count, tile_idx)

                for pred_box in pred_boxes:
                    pred_box.xmin += tile_bbox.xmin
                    pred_box.xmax += tile_bbox.xmin
                    pred_box.ymin += tile_bbox.ymin
                    pred_box.ymax += tile_bbox.ymin

                    corrected_boxes += [pred_box]

            return corrected_boxes

    def timed_infer_image(self, image):
        tile_count = self.model_config.get('tiles', 1)

        result = None
        time_stat = {
            'infer': 0,
            'correction': 0,
            'im_preproc': 0,
            'net_preproc': 0,
            'net_postproc': 0,
            'correct_n_nms': 0,
            'raw_boxes_count': 0,
            'out_boxes_count': 0
        }

        start_time = time.time()
        if tile_count == 1:
            images_batch = np.expand_dims(image, axis=0)
        else:
            images_batch = utils.tiles_image2batch(image, tile_count)
        time_stat['im_preproc'] = time.time() - start_time

        start_time = time.time()
        nb_images, image_h, image_w, image_c = images_batch.shape
        batch_input = np.zeros((nb_images, self.infer_sz[0], self.infer_sz[1], image_c))

        for i, im_batch_smpl in enumerate(images_batch):
            batch_input[i] = utils.image2net_input_sz(im_batch_smpl, self.infer_sz[0], self.infer_sz[1])

        batch_input = utils.image_normalize(batch_input)
        time_stat['net_preproc'] = time.time() - start_time

        start_time = time.time()
        batch_output = self.infer_model.predict_on_batch(batch_input)
        time_stat['infer'] = time.time() - start_time

        start_time = time.time()
        batch_boxes = [None] * nb_images
        net_output_count = len(batch_output)

        for i in range(nb_images):
            yolos = [batch_output[b][i] for b in range(net_output_count)]
            boxes = []

            for j in range(len(yolos)):
                l_idx = net_output_count - 1
                r_idx = net_output_count

                yolo_anchors = self.anchors[(l_idx - j) * 6:(r_idx - j) * 6]
                boxes += utils.decode_netout(yolos[j], yolo_anchors, self.obj_thresh, self.infer_sz[0], self.infer_sz[1])

            time_stat['raw_boxes_count'] += len(boxes)
            start_time_in = time.time()
            boxes = utils.correct_yolo_boxes(boxes, image_h, image_w, self.infer_sz[0], self.infer_sz[1])
            batch_boxes[i] = utils.do_nms(boxes, self.nms_thresh)

            time_stat['correct_n_nms'] += time.time() - start_time_in
            time_stat['out_boxes_count'] += len(boxes)

        pred_batch_boxes = batch_boxes
        time_stat['net_postproc'] = time.time() - start_time

        start_time = time.time()
        if tile_count == 1:
            result = pred_batch_boxes[0]
        else:
            corrected_boxes = []
            for tile_idx, pred_boxes in enumerate(pred_batch_boxes):
                tile_bbox = utils.tiles_get_bbox(image.shape[0:2], tile_count, tile_idx)

                for pred_box in pred_boxes:
                    pred_box.xmin += tile_bbox.xmin
                    pred_box.xmax += tile_bbox.xmin
                    pred_box.ymin += tile_bbox.ymin
                    pred_box.ymax += tile_bbox.ymin

                    corrected_boxes += [pred_box]

            result = corrected_boxes
        time_stat['correction'] = time.time() - start_time

        return result, time_stat

    def _infer_on_batch(self, images, return_boxes=True):
        nb_images, image_h, image_w, image_c = images.shape
        batch_input = np.zeros((nb_images, self.infer_sz[0], self.infer_sz[1], image_c))

        for i in range(nb_images):
            batch_input[i] = utils.image2net_input_sz(images[i], self.infer_sz[0], self.infer_sz[1])

        batch_input = utils.image_normalize(batch_input)

        batch_output = self.infer_model.predict_on_batch(batch_input)
        batch_boxes = [None] * nb_images
        if not return_boxes:
            return None

        net_output_count = len(batch_output)

        for i in range(nb_images):
            # This is required for one-output networks!
            if net_output_count > 1:
                yolos = [batch_output[b][i] for b in range(net_output_count)]
            else:
                yolos = [batch_output[i]]

            boxes = []

            for j in range(len(yolos)):
                l_idx = net_output_count - 1
                r_idx = net_output_count

                yolo_anchors = self.anchors[(l_idx - j) * 6:(r_idx - j) * 6]
                boxes += utils.decode_netout(yolos[j], yolo_anchors, self.obj_thresh, self.infer_sz[0], self.infer_sz[1])

            # for yolo_bbox in boxes:
                # print("Before NMS: {}".format(yolo_bbox))

            boxes = utils.correct_yolo_boxes(boxes, image_h, image_w, self.infer_sz[0], self.infer_sz[1])
            boxes = utils.do_nms(boxes, self.nms_thresh)

            # for yolo_bbox in boxes:
                # print("After NMS: {}",format(yolo_bbox.get_str()))

            batch_boxes[i] = boxes

        return batch_boxes

    def evaluate_generator(self,
                           generator,
                           iou_threshold=0.5,
                           save_path=None,
                           verbose=False):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            model           : The model to evaluate.
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            obj_thresh      : The threshold used to distinguish between object and non-object
            nms_thresh      : The threshold used to determine whether two detections are duplicates
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        iterator = range(generator.size())
        if verbose:
            # Modify with rendering
            iterator = tqdm(iterator)

        _sum_inference_time = 0
        _inference_cnt = 0

        for i in iterator:
            raw_image = generator.load_full_image(i)

            # make the boxes and the labels
            _start_inf_time = time.time()
            pred_boxes = self.infer_image(raw_image)
            _sum_inference_time += time.time() - _start_inf_time
            _inference_cnt += 1

            score = np.array([box.get_best_class_score() for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array(
                    [[box.xmin, box.ymin, box.xmax, box.ymax, box.get_best_class_score()]
                        for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_full_annotation_bboxes(i)

            for label in range(generator.num_classes()):
                all_annotations[i][label] = \
                    np.array([[box.xmin, box.ymin, box.xmax, box.ymax]
                              for box in annotations if box.class_idx == label])

#             for pred_box in pred_boxes:
#                 print(pred_box)

            # print('Annotations:')
            # for annot_image in all_annotations:
            #     for label, annot in enumerate(annot_image):
            #         if annot is not None and len(annot) > 0:
            #             print('Annotation: {} / {}'.format(label, annot))

            # print('Detections:')
            # for det_image in all_detections:
            #     for label, detec in enumerate(det_image):
            #         if detec is not None and len(detec) > 0:
            #             print('Detection: {} / {}'.format(label, detec))

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label_idx in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            iterator = range(generator.size())
            if verbose:
                print('Processing label: {} - {}'.format(label_idx, generator.get_class_name(label_idx)))
                # Modify with rendering
                iterator = tqdm(iterator)

            for i in iterator:
                detections = all_detections[i][label_idx]
                annotations = all_annotations[i][label_idx]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                if save_path:
                    fpath = os.path.join(save_path, 'det_{}.png'.format(i))
                    render_image = generator.load_image(i)
                    for annot in annotations:
                        cv2.rectangle(render_image,
                                    (annot[0], annot[1]),
                                    (annot[2], annot[3]),
                                    (0, 255, 0), 3)

                for d in detections:
                    scores = np.append(scores, d[4])

                    if save_path:
                        cv2.rectangle(render_image,
                                    (int(d[0]), int(d[1])),
                                    (int(d[2]), int(d[3])),
                                    (255, 0, 0), 2)

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = utils.compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

                if save_path:
                    cv2.imwrite(fpath, render_image)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label_idx] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / \
                np.maximum(true_positives + false_positives,
                        np.finfo(np.float64).eps)

            # compute average precision
            average_precision = utils.compute_ap(recall, precision)
            average_precisions[generator.get_class_name(label_idx)] = average_precision

        return average_precisions, (_sum_inference_time / _inference_cnt)
