import cv2
import copy
import numpy as np
from utils.utils import preprocess_input
from tensorflow.keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes


class BatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 anchors,
                 labels,
                 min_net_size=[0, 0],
                 max_net_size=[0, 0],
                 downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                 max_box_per_image=30,
                 batch_size=1,
                 shuffle=True,
                 jitter=0.3,
                 scale_distr=0.25,
                 flip=False,
                 norm=None,
                 infer_sz=None,
                 mem_mode=True
                 ):
        self.instances = instances
        self.batch_size = batch_size
        self.labels = labels
        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.shuffle = shuffle
        self.mem_mode = mem_mode

        self.min_net_size = (np.array(min_net_size) //
                             self.downsample) * self.downsample
        self.max_net_size = (np.array(max_net_size) //
                             self.downsample) * self.downsample

        self.scale_distr = scale_distr
        self.jitter = jitter
        self.flip = flip
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1])
                        for i in range(len(anchors) // 2)]
        self.infer_sz = infer_sz

        self.net_size_h = 0
        self.net_size_w = 0

        self.anchors_per_output = 3
        self.output_layers_count = len(self.anchors) // self.anchors_per_output

        if self.output_layers_count not in [1, 2, 3]:
            assert False, "Unchecked network output"

        if shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):

        if self.infer_sz:
            net_h, net_w = self.infer_sz
        else:
            # get image input size, change every 10 batches
            net_h, net_w = self._get_net_size(idx)

        # base_grid_h, base_grid_w = int((net_h + self.downsample / 2) // self.downsample),
        #                            int((net_w + self.downsample / 2) // self.downsample)
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros(
            (r_bound - l_bound, net_h, net_w, 3))  # input images
        # list of groundtruth boxes
        t_batch = np.zeros(
            (r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))

        dummies = [np.zeros((r_bound - l_bound, 1))
                   for i in range(self.output_layers_count)]

        # According to reversed outputs
        yolos = [np.zeros((r_bound - l_bound,
                           (2 ** i) * base_grid_h,
                           (2 ** i) * base_grid_w,
                           self.anchors_per_output,
                           4 + 1 + len(self.labels)))
                 for i in reversed(range(self.output_layers_count))]

        # yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//self.output_layers_count, 4+1+len(self.labels))) # desired network output 1
        # yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//self.output_layers_count, 4+1+len(self.labels))) # desired network output 2
        # yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//self.output_layers_count, 4+1+len(self.labels))) # desired network output 3
        # yolos = [yolo_3, yolo_2, yolo_1]

        # if self.infer_sz:
        #     print(net_h, net_w)
        #     print(base_grid_h, base_grid_w)
        #     print(yolos[0].shape)

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)

            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       obj['xmax'] - obj['xmin'],
                                       obj['ymax'] - obj['ymin'])

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
                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) /
                           float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) /
                           float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, output_anchor_idx] = 0
                yolo[instance_count, grid_y, grid_x,
                     output_anchor_idx, 0:4] = box
                yolo[instance_count, grid_y, grid_x, output_anchor_idx, 4] = 1.
                yolo[instance_count, grid_y, grid_x,
                     output_anchor_idx, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] -
                            obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

                # assign input image to x_batch
            if self.norm:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(
                        img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 255, 0), 2)
                    cv2.putText(img, obj['name'],
                                (obj['xmin']+2, obj['ymin']+2),
                                0, 2e-3 * img.shape[0],
                                (255, 0, 0), 2)

#                     print(train_instance['filename'])
#                     print(obj['name'])

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        if self.norm:
            return [x_batch, t_batch] + [yolo for yolo in reversed(yolos)], dummies
        else:
            return x_batch

    def _get_net_size(self, idx):
        if idx % 10 == 0 or self.net_size_w == 0:
            # rate = float(self.max_net_size[1]) / self.max_net_size[0]

            # new_h = self.downsample * np.random.randint(self.min_net_size[0] / self.downsample - 2,
            #                                             self.max_net_size[0] / self.downsample + 2 + 1)

            # new_w = new_h * rate

            self.net_size_h = self.downsample * np.random.randint(self.min_net_size[0] / self.downsample,
                                                                  self.max_net_size[0] / self.downsample + 1)
            self.net_size_w = self.downsample * np.random.randint(self.min_net_size[1] / self.downsample,
                                                                  self.max_net_size[1] / self.downsample + 1)

            # self.net_size_h = int(new_h)
            # self.net_size_w = int(new_w)

        return self.net_size_h, self.net_size_w

    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None:
            print('Cannot find ', image_name)

        # image = image[:,:,::-1] # RGB image
        image_h, image_w, _ = image.shape
        flip = 0

        if self.infer_sz:
            new_ar = image_w * 1. / image_h

            if new_ar < 1:
                new_h = int(net_h)
                new_w = int(net_h * new_ar)
            else:
                new_w = int(net_w)
                new_h = int(net_w / new_ar)

            dx = int((net_w - new_w) // 2)
            dy = int((net_h - new_h) // 2)

            # apply scaling and cropping
            im_sized = apply_random_scale_and_crop(
                image, new_w, new_h, net_w, net_h, dx, dy)

        else:
            # determine the amount of scaling and cropping
            dw = self.jitter * image_w
            dh = self.jitter * image_h

            new_ar = (image_w + np.random.uniform(-dw, dw)) / \
                (image_h + np.random.uniform(-dh, dh))
            scale = np.random.uniform(
                1 - self.scale_distr, 1 + self.scale_distr)

            if new_ar < 1:
                new_h = int(scale * net_h)
                new_w = int(net_h * new_ar)
            else:
                new_w = int(scale * net_w)
                new_h = int(net_w / new_ar)

            dx = int(np.random.uniform(0, net_w - new_w))
            dy = int(np.random.uniform(0, net_h - new_h))

            # apply scaling and cropping
            im_sized = apply_random_scale_and_crop(
                image, new_w, new_h, net_w, net_h, dx, dy)

            # randomly distort hsv space
            im_sized = random_distort_image(
                im_sized, hue=18, saturation=1.1, exposure=1.1)

            # randomly flip
            if self.flip:
                flip = np.random.randint(2)
                im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(
            instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)

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

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'],
                     obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])
