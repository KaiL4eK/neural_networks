from keras.regularizers import l2
from functools import wraps, reduce
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers.merge import add, concatenate
import backend
import os

from keras.layers import Concatenate, MaxPooling2D
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU
from keras.layers import ZeroPadding2D, UpSampling2D, Lambda, Conv2DTranspose, Flatten, ReLU


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))


class YoloLayer(Layer):
    def __init__(self, anchors, max_grid_hw, batch_size, warmup_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, debug=False,
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(
            anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale

        print('YoloLayer anchors: {} / max_grid: {}'.format(anchors, max_grid_hw))

        self.debug = debug
        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid_hw

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [
                             max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))

        cell_y = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_h), [
                             max_grid_w]), (1, max_grid_w, max_grid_h, 1, 1)))
        cell_y = tf.transpose(cell_y, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(YoloLayer, self).build(input_shape)

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat(
            [tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(
            tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(
            tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

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
        true_box_class = tf.argmax(y_true[..., 5:], -1)

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
            tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

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
        conf_delta *= tf.expand_dims(tf.to_float(best_ious <
                                                 self.ignore_thresh), 4)

        """
        Compute some online statistics
        """
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

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
        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask = tf.expand_dims(tf.to_float(
            tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50 = tf.reduce_sum(tf.to_float(
            iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = tf.reduce_sum(tf.to_float(
            iou_scores >= 0.75) * detect_mask * class_mask) / (count + 1e-3)
        avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
        avg_noobj = tf.reduce_sum(
            pred_box_conf * (1-object_mask)) / (count_noobj + 1e-3)
        avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1),
                                                      lambda: [true_box_xy + (0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (1-object_mask),
                                                               true_box_wh +
                                                               tf.zeros_like(true_box_wh) *
                                                               (1-object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(
            2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)

        xy_delta = xywh_mask * (pred_box_xy-true_box_xy) * \
            wh_scale * self.xywh_scale
        wh_delta = xywh_mask * (pred_box_wh-true_box_wh) * \
            wh_scale * self.xywh_scale
        conf_delta = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + \
            (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
            tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
            self.class_scale

        loss_xy = tf.reduce_sum(tf.square(xy_delta),       list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta),       list(range(1, 5)))
        loss_conf = tf.reduce_sum(tf.square(conf_delta),     list(range(1, 5)))
        loss_class = tf.reduce_sum(
            class_delta,               list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if self.debug:
            loss = tf.Print(loss, [grid_h, avg_obj],
                            message='avg_obj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_noobj],
                            message='avg_noobj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_iou],
                            message='avg_iou \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_cat],
                            message='avg_cat \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall50],
                            message='recall50 \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall75],
                            message='recall75 \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, count],
                            message='count \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
                                   tf.reduce_sum(loss_wh),
                                   tf.reduce_sum(loss_conf),
                                   tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def create_squeeze_model(
    nb_class,
    anchors,
    max_box_per_image,
    max_grid,
    batch_size,
    warmup_batches,
    ignore_thresh,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    train_shape
):
    outputs = 1
    anchors_per_output = len(anchors)//2//outputs

    image_input = Input(shape=train_shape, name='input_img')
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4),
                       name='input_true_boxes')
    # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_1 = Input(shape=(None, None, anchors_per_output,
                               4+1+nb_class), name='input_true_yolo_x32')

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2 *
                                 anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        # x     = LeakyReLU(name=s_id + relu + sq1x1)(x)
        x = ReLU(6., name=s_id + relu + sq1x1)(x)

        left = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
        # left  = LeakyReLU(name=s_id + relu + exp1x1)(left)
        left = ReLU(6., name=s_id + relu + exp1x1)(left)

        right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
        # right = LeakyReLU(name=s_id + relu + exp3x3)(right)
        right = ReLU(6., name=s_id + relu + exp3x3)(right)

        x = add([left, right], name=s_id + 'concat')

        return x

    x = Conv2D(64, (3, 3), strides=(2, 2),
               padding='same', name='conv1')(image_input)
    # x = LeakyReLU(name='relu_conv1')(x)
    x = ReLU(6., name='relu_conv1')(x)

    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)

    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)

    x = MaxPooling2D(pool_size=(2, 2), name='pool5')(x)

    # x = ZeroPadding2D(padding=(1, 1))(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)

    x = MaxPooling2D(pool_size=(2, 2), name='pool7')(x)

    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    pred_yolo_1 = Conv2D(pred_filter_count,
                         (1, 1),
                         strides=(1, 1),
                         padding='same',
                         name='DetectionLayer',
                         kernel_initializer='lecun_normal')(x)

    loss_yolo_1 = YoloLayer(yolo_anchors[0],
                            [1*num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([image_input, pred_yolo_1, true_yolo_1, true_boxes])

    train_model = Model([image_input, true_boxes, true_yolo_1], [loss_yolo_1])
    infer_model = Model(image_input, [pred_yolo_1])

    return train_model, infer_model, infer_model, 0


def create_xception_model(
    nb_class,
    anchors,
    max_box_per_image,
    max_grid,
    batch_size,
    warmup_batches,
    ignore_thresh,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    train_shape,
    **kwargs
):
    outputs = 1
    anchors_per_output = len(anchors)//2//outputs

    image_input = Input(shape=train_shape, name='input_img')
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4),
                       name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class),
                        name='input_true_yolo_x32')  # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2 *
                                 anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    from keras.applications.xception import Xception

    xception = Xception(input_tensor=image_input,
                        include_top=False, weights='imagenet')

    x = xception.output

    pred_yolo_1 = Conv2D(pred_filter_count, 1, padding='same',
                         strides=1, name='DetectionLayer1')(x)
    loss_yolo_1 = YoloLayer(yolo_anchors[0],
                            [1*num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([image_input, pred_yolo_1, true_yolo_1, true_boxes])

    train_model = Model([image_input, true_boxes, true_yolo_1], [loss_yolo_1])
    infer_model = Model(image_input, [pred_yolo_1])

    freeze_layers_cnt = len(infer_model.layers) - 4
    print('Non-freezed layers {}'.format(freeze_layers_cnt))

    mvnc_model = infer_model

    return train_model, infer_model, mvnc_model, freeze_layers_cnt


def create_yolo_head_models(
    done_backend,
    nb_class,
    anchors,
    anchors_per_output,
    max_input_size,
    max_box_per_image=1,
    batch_size=1,
    warmup_batches=0,
    ignore_thresh=0.5,
    grid_scales=[1, 1, 1],
    obj_scale=1,
    noobj_scale=1,
    xywh_scale=1,
    class_scale=1,
):
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4),
                       name='input_true_boxes')
    true_yolos = []
    for i in range(len(done_backend.pred_layers)):
        # grid_h, grid_w, nb_anchor, 5+nb_class
        true_yolo = Input(shape=(None, None, anchors_per_output, 4+1+nb_class),
                          name='input_true_yolo_{}_x{}'.format(i, done_backend.downgrades[i]))

        true_yolos += [true_yolo]

    yolo_anchors = []
    for i in reversed(range(len(done_backend.pred_layers))):
        yolo_anchors += [anchors[i*2 *
                                 anchors_per_output:(i+1)*2*anchors_per_output]]

    image_input = done_backend.input_layers[0]
    loss_yolos = []
    for idx, pred in enumerate(done_backend.pred_layers):
        max_grid_hw = np.array([max_input_size[0] // done_backend.downgrades[idx],
                                max_input_size[1] // done_backend.downgrades[idx]])

        loss_yolo = YoloLayer(yolo_anchors[idx],
                              max_grid_hw,
                              batch_size,
                              warmup_batches,
                              ignore_thresh,
                              grid_scales[idx],
                              obj_scale,
                              noobj_scale,
                              xywh_scale,
                              class_scale)([image_input, pred, true_yolos[idx], true_boxes])

        loss_yolos += [loss_yolo]

    train_model = Model([image_input, true_boxes] + true_yolos, loss_yolos)
    infer_model = Model(image_input, done_backend.pred_layers)
    mvnc_model = infer_model

    return train_model, infer_model, mvnc_model


def create_model_new(
    nb_class,
    anchors,
    max_input_size,
    anchors_per_output,
    max_box_per_image=1,
    batch_size=1,
    base='Tiny',
    warmup_batches=0,
    ignore_thresh=0.5,
    multi_gpu=1,
    grid_scales=[1, 1, 1],
    obj_scale=1,
    noobj_scale=1,
    xywh_scale=1,
    class_scale=1,
    train_shape=(None, None, 3),
    load_src_weights=True,
    is_freezed=False
):
    backend_options = {
        'pred_filters':         anchors_per_output*(4+1+nb_class),
        'train_shape':          train_shape
    }

    backends = {'TinyV3':           (backend.Tiny_YOLOv3,       "yolov3-tiny.h5"),
                'Darknet53':        (backend.Darknet53,         "yolov3_exp.h5"),
                'Darknet19':        (backend.Darknet19,         "yolov2.h5"),
                'MobileNetv2_35':   (backend.MobileNetV2_35,    ""),
                'MobileNetv2_50':   (backend.MobileNetV2_50,    ""),
                'MobileNetv2_75':   (backend.MobileNetV2_75,    ""),
                'MobileNetv2_100':  (backend.MobileNetV2_100,   ""),
                'SqueezeNet':       (create_squeeze_model,      ""),
                'Xception':         (create_xception_model,     "")
                }

    if base not in backends:
        print('No such base: {}'.format(base))
        return None

    print('Loading "{}" model'.format(base))

    new_backend = backends[base][0](**backend_options)
    train_model, infer_model, mvnc_model = create_yolo_head_models(
        new_backend,
        nb_class,
        anchors,
        anchors_per_output,
        max_input_size,
        max_box_per_image,
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
    )

    if backends[base][1]:
        print('Loading {}'.format(backends[base][1]))
        train_model.load_weights(os.path.join(
            'src_weights', backends[base][1]), by_name=True, skip_mismatch=True)

    if is_freezed and new_backend.head_layers_cnt > 0:
        freeze_layers_cnt = len(infer_model.layers) - \
            new_backend.head_layers_cnt

        print('Freezing %d layers of %d' %
              (freeze_layers_cnt, len(infer_model.layers)))
        for l in range(freeze_layers_cnt):
            infer_model.layers[l].trainable = False

    return train_model, infer_model, mvnc_model
