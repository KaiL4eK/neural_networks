import cv2
import numpy as np
from keras.utils import to_categorical, Sequence

import imgaug as ia
from imgaug import augmenters as iaa

print(cv2.useOptimized())

class BatchGenerator(Sequence):
    def __init__(self, 
        instances,
        labels,
        input_sz,
        batch_size=1,
        shuffle=True, 
        jitter=True, 
        norm=None,
        infer=None
    ):
        self.batch_size         = batch_size
        self.labels             = labels

        # Convert instances to pairs
        self.instances          = []
        for c_name, img_fpaths in instances.items():
            for img_fpath in img_fpaths:
                c_index = labels.index(c_name)
                self.instances += [(c_index, img_fpath)]

        if not infer:
            print('Size of train dataset: {}'.format(len(self.instances)))
        else:
            print('Size of inference dataset: {}'.format(len(self.instances)))

        self.input_sz           = input_sz
        self.infer              = infer

        self.shuffle            = shuffle
        self.jitter             = jitter
        self.norm               = norm

        if shuffle: 
            np.random.shuffle(self.instances)
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))

    def __getitem__(self, idx):
        net_h, net_w = self._get_net_size()

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))    # input images
        y_batch = np.zeros((r_bound - l_bound, len(self.labels)))   # list of groundtruth

        instance_count = 0

        # do the logic to fill in the inputs and the output
        for c_idx, img_fpath in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img = self._aug_image(img_fpath, net_h, net_w)

            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                x_batch[instance_count] = img

            y_batch[instance_count] = to_categorical(c_idx, len(self.labels))

            # increase instance counter in the current batch
            instance_count += 1                 
        
        if self.norm is not None:
            return x_batch, y_batch
        else:
            return x_batch

    def _get_net_size(self):
        return self.input_sz, self.input_sz

    def _aug_image(self, img_fpath, net_h, net_w):
        image = cv2.imread(img_fpath) # RGB image
        
        if image is None:
            print('Cannot find ', img_fpath)
        
        # image = image[:,:,::-1] # RGB image
        image_h, image_w, _ = image.shape

        im_sized = cv2.resize(image, (net_w, net_h), interpolation=cv2.INTER_LINEAR)

        if self.infer:
            return im_sized
        else:
            pass

        return im_sized

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