from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda, Conv2DTranspose, Flatten
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.engine.topology import Layer
import tensorflow as tf

class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, debug=False, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        print('YoloLayer anchors: {} / max_grid: {}'.format(anchors, max_grid))

        self.debug          = debug
        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
        pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      

        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = y_true[..., 2:4] # t_wh
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)         

        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half      

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        count       = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)    
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
        avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) 

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """      
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + \
                        (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if self.debug:
            loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
            loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)   
            loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)     
            loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy), 
                                           tf.reduce_sum(loss_wh), 
                                           tf.reduce_sum(loss_conf), 
                                           tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)   


        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]

def create_yolo_squeeze_model(
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
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x32') # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2*anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    from keras.layers import MaxPooling2D

    sq1x1  = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu   = "relu_"

    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x     = LeakyReLU(name=s_id + relu + sq1x1)(x)

        left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left  = LeakyReLU(name=s_id + relu + exp1x1)(left)

        right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
        right = LeakyReLU(name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=3, name=s_id + 'concat')

        return x

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(image_input)
    x = LeakyReLU(name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool7')(x)

    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    pred_yolo_1 = Conv2D(pred_filter_count, 
                        (1,1),
                        strides=(1,1), 
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

    return [train_model, infer_model]


def create_yolov2_model(
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
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x32') # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2*anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    from keras.layers import MaxPooling2D
    from keras.layers.merge import concatenate

    def space_to_depth_x2(x):
        import tensorflow as tf
        return tf.space_to_depth(x, block_size=2)

    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(image_input)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Backend end =)

    pred_yolo_1 = Conv2D(pred_filter_count, 
                        (1,1),
                        strides=(1,1), 
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

    return [train_model, infer_model]

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
    train_shape
):
    outputs = 1
    anchors_per_output = len(anchors)//2//outputs

    image_input = Input(shape=train_shape, name='input_img')
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x32') # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2*anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    from keras.applications.xception import Xception

    xception = Xception(input_tensor=image_input, include_top=False, weights='imagenet')

    x = xception.output


    pred_yolo_1 = Conv2D(pred_filter_count, 1, padding='same', strides=1, name='DetectionLayer1')(x)
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

    return [train_model, infer_model]



def create_mobilenetv2_model(
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
    outputs = 2
    anchors_per_output = len(anchors)//2//outputs

    image_input = Input(shape=train_shape, name='input_img')
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x32') # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x16') # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2*anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    from keras.applications.mobilenetv2 import MobileNetV2

    mobilenetv2 = MobileNetV2(input_tensor=image_input, include_top=False, weights='imagenet')

    out13 = mobilenetv2.output

    pred_yolo_1 = Conv2D(pred_filter_count, 1, padding='same', strides=1, name='DetectionLayer1')(out13)
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


    x = Conv2D(256, 1, strides=(1,1), padding='same', name='conv_20', use_bias=False)(out13)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    from keras.layers import Conv2DTranspose

    x = Conv2DTranspose(256, 1, strides=2, padding='same')(x)

    out26 = mobilenetv2.get_layer(name='block_13_expand_relu').output

    x = concatenate([x, out26])

    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(x)
    x = BatchNormalization(name='norm_21')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_23', use_bias=False)(x)
    x = BatchNormalization(name='norm_23')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_24', use_bias=False)(x)
    x = BatchNormalization(name='norm_24')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_25', use_bias=False)(x)
    x = BatchNormalization(name='norm_25')(x)
    x = LeakyReLU(alpha=0.1)(x)

    pred_yolo_2 = Conv2D(pred_filter_count, 1, padding='same', strides=1, name='DetectionLayer2')(x)
    loss_yolo_2 = YoloLayer(yolo_anchors[1], 
                        [2*num for num in max_grid], 
                        batch_size, 
                        warmup_batches, 
                        ignore_thresh, 
                        grid_scales[1],
                        obj_scale,
                        noobj_scale,
                        xywh_scale,
                        class_scale)([image_input, pred_yolo_2, true_yolo_2, true_boxes])

    train_model = Model([image_input, true_boxes, true_yolo_1, true_yolo_2], [loss_yolo_1, loss_yolo_2])
    infer_model = Model(image_input, [pred_yolo_1, pred_yolo_2])

    return [train_model, infer_model]


def create_yolov3_model(
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
    outputs = 3
    anchors_per_output = len(anchors)//2//outputs

    image_input = Input(shape=train_shape, name='input_img')
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x32') # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x16') # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x8') # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []
    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    
    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2*anchors_per_output:(i+1)*2*anchors_per_output]]

    
    pred_filter_count = (anchors_per_output*(5+nb_class))

    def _conv_block(inp, convs, do_skip=True):
        x = inp
        count = 0
        
        for conv in convs:
            if count == (len(convs) - 2) and do_skip:
                skip_connection = x
            count += 1
            
            if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # unlike tensorflow darknet prefer left and top paddings
            x = Conv2D(conv['filter'], 
                       conv['kernel'], 
                       strides=conv['stride'], 
                       padding='valid' if conv['stride'] > 1 else 'same', # unlike tensorflow darknet prefer left and top paddings
                       name='conv_' + str(conv['layer_idx']), 
                       use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

        return add([skip_connection, x]) if do_skip else x        


    # Layer  0 => 4
    x = _conv_block(image_input, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], do_skip=False)

    # Layer 80 => 82
    pred_yolo_1 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                             {'filter': pred_filter_count, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], do_skip=False)
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

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], do_skip=False)

    # Layer 92 => 94
    pred_yolo_2 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                             {'filter': pred_filter_count, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], do_skip=False)
    loss_yolo_2 = YoloLayer(yolo_anchors[1], 
                            [2*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([image_input, pred_yolo_2, true_yolo_2, true_boxes])

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    pred_yolo_3 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                             {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                             {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                             {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                             {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                             {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                             {'filter': pred_filter_count, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], do_skip=False)
    loss_yolo_3 = YoloLayer(yolo_anchors[2], 
                            [4*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([image_input, pred_yolo_3, true_yolo_3, true_boxes]) 

    train_model = Model([image_input, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = Model(image_input, [pred_yolo_1, pred_yolo_2, pred_yolo_3])

    return [train_model, infer_model]


def create_tiny_yolov3_model(
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
    outputs = 2
    anchors_per_output = len(anchors)//2//outputs

    image_input = Input(shape=train_shape, name='input_img')
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4), name='input_true_boxes')
    true_yolo_1 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x32') # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(shape=(None, None, anchors_per_output, 4+1+nb_class), name='input_true_yolo_x16') # grid_h, grid_w, nb_anchor, 5+nb_class

    yolo_anchors = []

    for i in reversed(range(outputs)):
        yolo_anchors += [anchors[i*2*anchors_per_output:(i+1)*2*anchors_per_output]]

    # yolo_anchors = [anchors[12:18], anchors[6:12], anchors[0:6]]
    pred_filter_count = (anchors_per_output*(5+nb_class))

    from keras.layers import Concatenate, MaxPooling2D
    from functools import wraps, reduce
    from keras.regularizers import l2

    @wraps(Conv2D)
    def DarknetConv2D(*args, **kwargs):
        """Wrapper to set Darknet parameters for Convolution2D."""
        darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
        darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
        darknet_conv_kwargs.update(kwargs)
        return Conv2D(*args, **darknet_conv_kwargs)

    def DarknetConv2D_BN_Leaky(*args, **kwargs):
        """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        return compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            BatchNormalization(),
            LeakyReLU(alpha=0.1))

    def compose(*funcs):
        """Compose arbitrarily many functions, evaluated left to right.

        Reference: https://mathieularose.com/function-composition-in-python/
        """
        # return lambda x: reduce(lambda v, f: f(v), funcs, x)
        if funcs:
            return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
        else:
            raise ValueError('Composition of empty sequence not supported.')


    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(image_input)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    pred_yolo_1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(pred_filter_count, (1,1)))(x2)

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

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            # UpSampling2D(2))(x2)
            Conv2DTranspose(filters=128, 
                            kernel_size=2, 
                            strides=2, 
                            padding='same', 
                            use_bias=False, 
                            kernel_regularizer=l2(5e-4)))(x2)
    pred_yolo_2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(pred_filter_count, (1,1)))([x2,x1])


    loss_yolo_2 = YoloLayer(yolo_anchors[1], 
                            [2*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([image_input, pred_yolo_2, true_yolo_2, true_boxes]) 

    train_model = Model([image_input, true_boxes, true_yolo_1, true_yolo_2], [loss_yolo_1, loss_yolo_2])
    infer_model = Model(image_input, [pred_yolo_1, pred_yolo_2])

    yolo1_flat = Flatten()(pred_yolo_1)
    yolo2_flat = Flatten()(pred_yolo_2)

    mvnc_output = Concatenate()([yolo1_flat, yolo2_flat])

    mvnc_model  = Model(image_input, mvnc_output)

    return [train_model, infer_model, mvnc_model]

def create_model(
    nb_class, 
    anchors, 
    max_box_per_image   = 1, 
    max_input_size      = 416, 
    batch_size          = 1, 
    base                = 'Tiny',
    warmup_batches      = 0, 
    ignore_thresh       = 0.5, 
    multi_gpu           = 1, 
    grid_scales         = [1, 1, 1],
    obj_scale           = 1,
    noobj_scale         = 1,
    xywh_scale          = 1,
    class_scale         = 1,
    train_shape         = (None, None, 3),
    load_src_weights    = True
):
    

    backends = {'Tiny':         (create_tiny_yolov3_model,  "src_weights/yolov3-tiny.h5",   2, 32),
                'Darknet53':    (create_yolov3_model,       "src_weights/yolov3_exp.h5",    3, 32),
                'Darknet19':    (create_yolov2_model,       "src_weights/yolov2.h5",        1, 32),
                'MobileNetv2':  (create_mobilenetv2_model,  "",                             22, 32),
                'SqueezeNet':   (create_yolo_squeeze_model, "",                             -1, 32),
                'Xception':     (create_xception_model,     "",                             1, 32)
                }

    max_grid = [max_input_size // backends[base][3], max_input_size // backends[base][3]]

    model_args = dict(  nb_class            = nb_class, 
                        anchors             = anchors, 
                        max_box_per_image   = max_box_per_image, 
                        max_grid            = max_grid, 
                        batch_size          = batch_size//multi_gpu, 
                        warmup_batches      = warmup_batches,
                        ignore_thresh       = ignore_thresh,
                        grid_scales         = grid_scales,
                        obj_scale           = obj_scale,
                        noobj_scale         = noobj_scale,
                        xywh_scale          = xywh_scale,
                        class_scale         = class_scale,
                        train_shape         = train_shape )

    anchor_count = len(anchors) // 2

    print('Loading "{}" model'.format(base))

    # if multi_gpu > 1:
    #     with tf.device('/cpu:0'):
    #         template_model, infer_model = backends[base][0](**model_args)
    # else:
    
    template_model, infer_model, mvnc_model = backends[base][0](**model_args)  

    orig_weights_name = backends[base][1]

    if load_src_weights and orig_weights_name:
        print("\nLoading original pretrained (%s)" % orig_weights_name)       
        template_model.load_weights(orig_weights_name, by_name=True, skip_mismatch=True) 

    from utils.multi_gpu_model import multi_gpu_model

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model      

    return train_model, infer_model, mvnc_model, backends[base][2]

def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))
