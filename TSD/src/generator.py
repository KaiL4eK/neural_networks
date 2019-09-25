import cv2
import copy
import numpy as np
import _common.utils as cutls
from tensorflow.keras.utils import Sequence
from _common.bbox import BoundBox, bbox_iou
from _common.image import apply_random_scale_and_crop, random_distort_image, random_flip


class BatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 anchors,
                 labels,
                 downsample,  # ratio between network input's size and network output's size, 32 for YOLOv3
                 min_net_size=[416, 416],
                 max_net_size=[416, 416],
                 max_box_per_image=30,
                 batch_size=1,
                 shuffle=False,
                 jitter=0.3,
                 scale_distr=0.25,
                 flip=False,
                 norm=None,
                 infer_sz=None,
                 mem_mode=True,
                 tile_count=1
                 ):
        self.instances = instances
        self.batch_size = batch_size
        self.labels = labels
        self.downsample = downsample
        self.max_downsample = max(downsample)
        self.max_box_per_image = max_box_per_image
        self.shuffle = shuffle
        self.mem_mode = mem_mode
        self.tile_count = tile_count

        self.min_net_size = (np.array(min_net_size) //
                             self.max_downsample) * self.max_downsample
        self.max_net_size = (np.array(max_net_size) //
                             self.max_downsample) * self.max_downsample

        self.scale_distr = scale_distr
        self.jitter = jitter
        self.flip = flip
        
        self.infer_sz = infer_sz
        if self.infer_sz:
            self.jitter = 0.0
            self.scale_distr = 0.0
            self.flip = 0

        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1])
                        for i in range(len(anchors) // 2)]
        
        self.net_size_h = 0
        self.net_size_w = 0

        self.anchors_per_output = 3
        self.output_layers_count = len(self.anchors) // self.anchors_per_output

        self._bboxes_key = '__bboxes'
        self._tile_ann_bboxes_key = '__tile_annot_bboxes'
        self._tile_img_bboxes_key = '__tile_img_bboxes'

        ### Prepare BoundingBoxes ###
        for sample_idx in range(len(self.instances)):
            img_sz = (self.instances[sample_idx]['height'], self.instances[sample_idx]['width'])

            # Create base bboxes
            self.instances[sample_idx][self._bboxes_key] = []

            for obj in self.instances[sample_idx]['object']:
                annot_bbox = BoundBox(obj['xmin'], obj['ymin'], 
                                      obj['xmax'], obj['ymax'], 
                                      label_name=obj['name'], 
                                      label_idx=self.labels.index(obj['name']))

                self.instances[sample_idx][self._bboxes_key] += [annot_bbox]

            # Create tiled bboxes
            self.instances[sample_idx][self._tile_ann_bboxes_key] = [[] for _ in range(self.tile_count)]
            self.instances[sample_idx][self._tile_img_bboxes_key] = []

            for tile_idx in range(self.tile_count):
                tile_bbox = cutls.tiles_get_bbox(img_sz, self.tile_count, tile_idx)
                self.instances[sample_idx][self._tile_img_bboxes_key] += [tile_bbox]

                for obj in self.instances[sample_idx]['object']:
                    annot_bbox = BoundBox(obj['xmin'], obj['ymin'], 
                                          obj['xmax'], obj['ymax'], 
                                          label_name=obj['name'], 
                                          label_idx=self.labels.index(obj['name']))

                    if tile_bbox.intersect(annot_bbox) is None:
                        continue

                    # Correct annot tile bbox
                    annot_bbox.xmin -= tile_bbox.xmin
                    annot_bbox.xmax -= tile_bbox.xmin
                    annot_bbox.ymin -= tile_bbox.ymin
                    annot_bbox.ymax -= tile_bbox.ymin

                    self.instances[sample_idx][self._tile_ann_bboxes_key][tile_idx] += [annot_bbox]
        ### Prepare BoundingBoxes ###
        
        if self.output_layers_count not in [1, 2, 3]:
            assert False, "Unchecked network output"

        if self.shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size * self.tile_count))

    def get_inst_bounds(self, idx):
        return idx * self.batch_size, (idx + 1) * self.batch_size

    def __getitem__(self, idx):
        net_h, net_w = self._update_net_size(idx)

        # base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        # determine the first and the last indices of the batch
        l_bound, r_bound = self.get_inst_bounds(idx)

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))  # input images
        # list of groundtruth boxes
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))

        # [print(net_h // self.downsample[i], net_w // self.downsample[i]) for i in reversed(range(self.output_layers_count))]
        # According to reversed outputs - because of anchors
        yolos = [np.zeros((r_bound - l_bound,
                           net_h // self.downsample[i],
                           net_w // self.downsample[i],
                           self.anchors_per_output,
                           4 + 1 + len(self.labels)))
                 for i in reversed(range(self.output_layers_count))]

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for inst_idx in range(l_bound, r_bound):
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(inst_idx, net_h, net_w)

            for objbox in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       objbox.xmax - objbox.xmin,
                                       objbox.ymax - objbox.ymin)

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                output_idx = max_index // self.anchors_per_output
                output_anchor_idx = max_index % self.anchors_per_output

                # determine the yolo to be responsible for this bounding box
                yolo = yolos[output_idx]
                # [52, 26, 13]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5 * (objbox.xmin + objbox.xmax)
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (objbox.ymin + objbox.ymax)
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((objbox.xmax - objbox.xmin) /
                           float(max_anchor.xmax))  # t_w
                h = np.log((objbox.ymax - objbox.ymin) /
                           float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(objbox.class_name)
                # print(self.labels, objbox.class_name, obj_indx)

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, output_anchor_idx] = 0
                yolo[instance_count, grid_y, grid_x, output_anchor_idx, 0:4] = box
                yolo[instance_count, grid_y, grid_x, output_anchor_idx, 4] = 1.
                yolo[instance_count, grid_y, grid_x, output_anchor_idx, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, 
                            objbox.xmax - objbox.xmin, 
                            objbox.ymax - objbox.ymin]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

                # assign input image to x_batch
            if self.norm:
                # print(net_h, net_w, x_batch.shape, img.shape)
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for objbox in all_objs:
                    cv2.rectangle(
                        img, (objbox.xmin, objbox.ymin), (objbox.xmax, objbox.ymax), (255, 255, 0), 2)
                    cv2.putText(img, objbox.class_name,
                                (objbox.xmin+2, objbox.ymin+2),
                                0, 2e-3 * img.shape[0],
                                (255, 0, 0), 2)

#                     print(train_instance['filename'])
#                     print(obj['name'])

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        dummies = [np.zeros((r_bound - l_bound, 1))
                   for i in range(self.output_layers_count)]

        if self.norm:
            return [x_batch, t_batch] + [yolo for yolo in reversed(yolos)], dummies
        else:
            return x_batch

    def _get_net_size(self, idx):
        return self._update_net_size(idx)

    def _update_net_size(self, idx):
        if self.infer_sz:
            self.net_size_h, self.net_size_w = self.infer_sz
        elif idx % 10 == 0 or self.net_size_w == 0:
            net_size_h = self.max_downsample * np.random.randint(self.min_net_size[0] / self.max_downsample,
                                                                 self.max_net_size[0] / self.max_downsample + 1)
            net_size_w = self.max_downsample * np.random.randint(self.min_net_size[1] / self.max_downsample,
                                                                 self.max_net_size[1] / self.max_downsample + 1)
            self.net_size_h, self.net_size_w = net_size_h, net_size_w

        return self.net_size_h, self.net_size_w

    def _aug_image(self, idx, net_h, net_w):
        image = self._load_image(idx)

        if image is None:
            print('Cannot find ', image_name)

        # image = image[:,:,::-1] # RGB image
        image_h, image_w, _ = image.shape
        flip = 0
            
        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / \
                    (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(1 - self.scale_distr,
                                    1 + self.scale_distr)

        if new_ar < 1:
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)

        dx = int((net_w - new_w)/2)
        dy = int((net_h - new_h)/2)
        if self.infer_sz is None:
            dx = int(np.random.uniform(0, net_w - new_w))
            dy = int(np.random.uniform(0, net_h - new_h))

        # print(new_w, new_h, net_w, net_h, image_h, image_w)

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        if self.infer_sz is None:
            # randomly distort hsv space
            im_sized = random_distort_image(im_sized, hue=18, saturation=1.1, exposure=1.1)

            # randomly flip
            if self.flip:
                flip = np.random.randint(2)
                im_sized = random_flip(im_sized, flip)

        # print('  Source image sz: {}x{}'.format(image_w, image_h))
        # print('  New image sz: {}x{}'.format(new_w, new_h))
        # print('  Correction to {}x{}'.format(net_w, net_h))
        # print('  Correected sz: {}x{}'.format(im_sized.shape[1], im_sized.shape[0]))

        # boxes = instance['object']
        boxes = self._load_annotation_bboxes(idx)
        
        # for objbox in boxes:
        #     print('    Bbox for {}: {}'.format(idx, objbox.get_str()))

        # correct the size and pos of bounding boxes
        all_objs = self._correct_bounding_boxes(boxes, new_w, new_h, net_h, net_w, dx, dy, flip, image_w, image_h)

        # for objbox in all_objs:
        #     print('    Corrected bbox for {}: {}'.format(idx, objbox.get_str()))

        return im_sized, all_objs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def get_class_name(self, label_idx):
        if label_idx >= self.num_classes() or label_idx < 0:
            return None

        return self.labels[label_idx]

    def size(self):
        return len(self.instances)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def _correct_bounding_boxes(self, boxes, new_w, new_h, net_h, net_w, dx, dy, flip, image_w, image_h):
        # randomize boxes' order
        np.random.shuffle(boxes)

        # correct sizes and positions
        sx, sy = float(new_w)/image_w, float(new_h)/image_h

        def _constrain(min_v, max_v, value):
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value 

        corrected_boxes = []

        for box in boxes:
            box.xmin = int(_constrain(0, net_w, box.xmin*sx + dx))
            box.xmax = int(_constrain(0, net_w, box.xmax*sx + dx))
            box.ymin = int(_constrain(0, net_h, box.ymin*sy + dy))
            box.ymax = int(_constrain(0, net_h, box.ymax*sy + dy))

            if box.xmax <= box.xmin or box.ymax <= box.ymin:
                continue

            if flip == 1:
                swap = box.xmin
                box.xmin = net_w - box.xmax
                box.xmax = net_w - swap

            corrected_boxes += [box]

        return corrected_boxes

    def load_full_image(self, i):
        img_idx = int(i / self.tile_count)
        return cv2.imread(self.instances[img_idx]['filename'])

    def load_full_annotation_bboxes(self, i):
        img_idx = int(i / self.tile_count)
        return copy.deepcopy(self.instances[img_idx][self._bboxes_key])

    def _load_image(self, i):
        img_idx = int(i / self.tile_count)
        tile_idx = i % self.tile_count

        image = self.load_full_image(i)
        tile_bbox = self.instances[img_idx][self._tile_img_bboxes_key][tile_idx]

        # print(i, tile_idx, tile_bbox.get_str())

        return image[tile_bbox.ymin:tile_bbox.ymax, tile_bbox.xmin:tile_bbox.xmax, :]

    def _load_annotation_bboxes(self, i):
        img_idx = int(i / self.tile_count)
        tile_idx = i % self.tile_count

        return copy.deepcopy(self.instances[img_idx][self._tile_ann_bboxes_key][tile_idx])
