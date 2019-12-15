import cv2
import numpy as np
import copy

def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1./scale

def random_flip(image, flip):
    if flip == 1:
        return cv2.flip(image, 1)
    return image

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:,:,1] *= dsat
    image[:,:,2] *= dexp

    # change hue
    image[:,:,0] += dhue
    image[:,:,0] -= (image[:,:,0] > 180)*180
    image[:,:,0] += (image[:,:,0] < 0)  *180

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))

    if dx > 0:
        im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:,-dx:,:]

    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)

    if dy > 0:
        im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:,:,:]

    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)

    return im_sized[:net_h, :net_w,:]
