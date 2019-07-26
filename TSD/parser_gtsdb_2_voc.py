import cv2
import os
import tqdm

import voc_generator as vc

import argparse
argparser = argparse.ArgumentParser(description='Help =)')
argparser.add_argument('-o', '--output', help='Result directory path')
argparser.add_argument('-r', '--root', help='Root dir of GTSDB')
args = argparser.parse_args()

imgs_path = args.root
annots_train_fpath = os.path.join(imgs_path, 'TrainIJCNN2013', 'gt.txt')
annots_full_fpath = os.path.join(imgs_path, 'FullIJCNN2013', 'gt.txt')

frames_train_root_dir = os.path.join(imgs_path, 'TrainIJCNN2013')
frames_full_root_dir = os.path.join(imgs_path, 'FullIJCNN2013')

train_frames_fpaths = [os.path.join(frames_train_root_dir, i)
                       for i in os.listdir(frames_train_root_dir)
                       if i.endswith('.ppm')]
full_frames_fpaths = [os.path.join(frames_full_root_dir, i)
                      for i in os.listdir(frames_full_root_dir)
                      if i.endswith('.ppm')]

dst_data_dir = args.output
dst_train_data_dir = dst_data_dir + '_Train'
dst_full_data_dir = dst_data_dir + '_Full'

class_dict = {
    0: 'speed limit 20 (prohibitory)',
    1: 'speed limit 30 (prohibitory)',
    2: 'speed limit 50 (prohibitory)',
    3: 'speed limit 60 (prohibitory)',
    4: 'speed limit 70 (prohibitory)',
    5: 'speed limit 80 (prohibitory)',
    6: 'restriction ends 80 (other)',
    7: 'speed limit 100 (prohibitory)',
    8: 'speed limit 120 (prohibitory)',
    9: 'no overtaking (prohibitory)',
    10: 'no overtaking (trucks) (prohibitory)',
    11: 'priority at next intersection (danger)',
    12: 'priority road (other)',
    13: 'give way (other)',
    14: 'stop (other)',
    15: 'no traffic both ways (prohibitory)',
    16: 'no trucks (prohibitory)',
    17: 'no entry (other)',
    18: 'danger (danger)',
    19: 'bend left (danger)',
    20: 'bend right (danger)',
    21: 'bend (danger)',
    22: 'uneven road (danger)',
    23: 'slippery road (danger)',
    24: 'road narrows (danger)',
    25: 'construction (danger)',
    26: 'traffic signal (danger)',
    27: 'pedestrian crossing (danger)',
    28: 'school crossing (danger)',
    29: 'cycles crossing (danger)',
    30: 'snow (danger)',
    31: 'animals (danger)',
    32: 'restriction ends (other)',
    33: 'go right (mandatory)',
    34: 'go left (mandatory)',
    35: 'go straight (mandatory)',
    36: 'go right or straight (mandatory)',
    37: 'go left or straight (mandatory)',
    38: 'keep right (mandatory)',
    39: 'keep left (mandatory)',
    40: 'roundabout (mandatory)',
    41: 'restriction ends (overtaking) (other)',
    42: 'restriction ends (overtaking (trucks)) (other)',
}

print('--- Copy frames to dst folder ---')

vc.copy_2_images_dir(dst_train_data_dir, train_frames_fpaths)
vc.copy_2_images_dir(dst_full_data_dir, full_frames_fpaths)


def read_annotations(annot_fpath):
    checked_files = {}

    with open(annot_fpath) as fp:
        for idx, line in enumerate(fp):
            fname = line.split(';')[0]
            xmin = int(line.split(';')[1])
            ymin = int(line.split(';')[2])
            xmax = int(line.split(';')[3])
            ymax = int(line.split(';')[4])
            classId = int(line.split(';')[5])

            info = (xmin, ymin, xmax, ymax, class_dict[classId])

            if fname in checked_files:
                checked_files[fname] += [info]
            else:
                checked_files[fname] = [info]

    return checked_files


def append_frames_wo_annot(checked_files, frames_fpath_list):
    for fpath in frames_fpath_list:
        fname = os.path.basename(fpath)
        if fname not in checked_files:
            checked_files[fname] = None


train_checked_files = read_annotations(annots_train_fpath)
full_checked_files = read_annotations(annots_full_fpath)

append_frames_wo_annot(train_checked_files, train_frames_fpaths)
append_frames_wo_annot(full_checked_files, full_frames_fpaths)

print('--- Creating annotations ---')

vc.generate_annotations(dst_train_data_dir, train_checked_files)
vc.generate_annotations(dst_full_data_dir, full_checked_files)
