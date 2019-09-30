import random
import argparse
import numpy as np

from _common.voc import parse_voc_annotation
import _common.utils as utils
import json


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))

    return sum/n


def run_kmeans(ann_dims, anchor_num):
    ann_num, anchor_dim = ann_dims.shape[0:2]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    # Get random indices to capture random points to set them as centroids
    indices = []
    for i in range(anchor_num):
        rnd = random.randrange(ann_num)

        while rnd in indices:
            rnd = random.randrange(ann_num)

        indices.append(rnd)

    # indices = [random.randrange(ann_num) for i in range(anchor_num)]

    # print(sorted(indices))
    centroids = ann_dims[indices]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            distances += [1 - IOU(ann_dims[i], centroids)]
        # distances.shape = (ann_num, anchor_num)
        distances = np.array(distances)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), dtype=np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]

        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

            if (centroid_sums[j] == 0).all():
                print('-----------------------')
                print('>>>', j)
                print(centroid_sums)
                print('-----------------------')
                print(indices)
                print('-----------------------')
                print(assignments)
                print('-----------------------')

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def print_anchors(centroids, img_size=(1, 1)):
    out_string = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices:
        # w, h
        out_string += str(int(anchors[i, 0]*img_size[1])) + \
            ',' + str(int(anchors[i, 1]*img_size[0])) + ', '

    print(out_string[:-2])


argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')
argparser.add_argument(
    '-a',
    '--anchors',
    default=9,
    help='number of anchors to use')

args = argparser.parse_args()


def _main_():
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    num_anchors = config['model']['anchors_per_output'] * len(config['model']['downsample'])
    # num_anchors = int(args.anchors)

    infer_sz = config['model']['infer_shape']

    train_imgs, train_labels = parse_voc_annotation(
        config['train']['annot_folder'],
        config['train']['image_folder'],
        config['train']['cache_name'],
        config['model']['labels']
    )

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        img_sz = (image['height'], image['width'])
        image_width = image['width']
        image_height = image['height']

        img_sz = utils.tiles_get_sz(img_sz, config['model']['tiles'])

        scale_rate = utils.get_embedded_img_scale(img_sz, infer_sz)

        for obj in image['object']:
            a_width = float(obj['xmax']) - float(obj['xmin'])
            a_height = float(obj["ymax"]) - float(obj['ymin'])

            new_obj_sz = (int(a_width * scale_rate), int(a_height * scale_rate))

            annotation_dims.append(tuple(map(float, new_obj_sz)))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for {} anchors: {:0.2f}'.format(num_anchors, avg_IOU(annotation_dims, centroids)))
    print_anchors(centroids)


if __name__ == '__main__':
    _main_()
