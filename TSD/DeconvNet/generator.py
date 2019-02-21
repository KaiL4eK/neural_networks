import cv2
import numpy as np
from keras.utils import to_categorical, Sequence

import imgaug as ia
from imgaug import augmenters as iaa


class BatchGenerator(Sequence):
    def __init__(self,
                 instances,
                 input_sz,
                 batch_size=1,
                 shuffle=True,
                 jitter=True,
                 norm=None,
                 infer=False,
                 downsample=1,
                 mem_mode=True
                 ):
        self.batch_size = batch_size
        self.downsample = downsample
        self.mem_mode = mem_mode

        # Convert instances to pairs
        if not self.mem_mode:
            self.instances = instances
        else:
            self.instances = []
            for ins in instances:
                self.instances += [(cv2.imread(ins[0]), cv2.imread(ins[1]))]

        if not infer:
            print('Size of train dataset: {}'.format(len(self.instances)))
        else:
            print('Size of inference dataset: {}'.format(len(self.instances)))

        self.input_sz = input_sz
        self.infer = infer

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        if shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):
        net_h, net_w = self._get_net_size()

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))  # input images
        y_batch = np.zeros((r_bound - l_bound, net_h//self.downsample, net_w//self.downsample, 2))  # list of groundtruth

        instance_count = 0

        # do the logic to fill in the inputs and the output
        for inst in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, msk = self._aug_image(inst, net_h, net_w)

            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                x_batch[instance_count] = img

            foreground_idxs = np.where((msk == [0, 255, 0]).all(axis=2))
            background_idxs = np.where((msk == [255, 0, 0]).all(axis=2))
            y_batch[instance_count][background_idxs] = [1, 0]
            y_batch[instance_count][foreground_idxs] = [0, 1]

            # cv2.imshow('back', y_batch[instance_count, :, :, 0])
            # cv2.imshow('front', y_batch[instance_count, :, :, 1])
            # cv2.waitKey(0)

            # increase instance counter in the current batch
            instance_count += 1

        if self.norm is not None:
            return x_batch, y_batch
        else:
            return x_batch

    def _get_net_size(self):
        return self.input_sz[1], self.input_sz[0]

    def _aug_image(self, inst, net_h, net_w):
        if self.mem_mode:
            image = inst[0]
            mask = inst[1]
        else:
            img_fpath, mask_fpath = inst
            image = cv2.imread(img_fpath)  # RGB image
            mask  = cv2.imread(mask_fpath)  # RGB image

            if image is None:
                print('Cannot find ', img_fpath)

            if mask is None:
                print('Cannot find ', mask_fpath)

        # image = image[:,:,::-1] # RGB image
        # image_h, image_w, _ = image.shape

        im_sized  = cv2.resize(image, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        msk_sized = cv2.resize(mask, (net_w//self.downsample, net_h//self.downsample), interpolation=cv2.INTER_NEAREST)

        if not self.infer:
            flip = np.random.randint(2)
            if flip == 1:
                im_sized = cv2.flip(im_sized, 1)
                msk_sized = cv2.flip(msk_sized, 1)

        return im_sized, msk_sized

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

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
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])
