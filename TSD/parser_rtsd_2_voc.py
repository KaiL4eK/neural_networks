import os
import tqdm
import pandas as pd

from shutil import copyfile

import voc_generator as vc

import argparse
argparser = argparse.ArgumentParser(description='Help =)')
argparser.add_argument('-r', '--root',
                       help='Root dir of RTSD detection')
argparser.add_argument('-o', '--output',
                       help='Result directory path')
argparser.add_argument('-s', '--show_frames',
                       action='store_true',
                       help='Show frames during saving')
args = argparser.parse_args()
print(args)

data_root_path = args.root

frames_dirs = ['rtsd-d1-frames', 'rtsd-d2-frames', 'rtsd-d3-frames']
gt_train_frame_fpaths = []
gt_test_frame_fpaths = []

for frames_dir in frames_dirs:
    frames_train_root_path = os.path.join(data_root_path, frames_dir, 'train')
    frames_test_root_path = os.path.join(data_root_path, frames_dir, 'test')

    gt_train_frame_fpaths += [os.path.join(frames_train_root_path, i)
                              for i in os.listdir(frames_train_root_path)]
    gt_test_frame_fpaths += [os.path.join(frames_test_root_path, i)
                             for i in os.listdir(frames_test_root_path)]


gt_paths = ['rtsd-d1-gt', 'rtsd-d2-gt', 'rtsd-d3-gt']
unique_classes = []

gt_train_csv_fname = 'train_gt.csv'
gt_test_csv_fname = 'test_gt.csv'

gt_train_annot_fpaths = []
gt_test_annot_fpaths = []


def pandas_get_unique_classes(csv_fpath):
    data = pd.read_csv(csv_fpath)
    unique_classes = data['sign_class'].unique().tolist()
    return unique_classes


for gt_path in gt_paths:
    gt_dirpath = os.path.join(data_root_path, gt_path)
    local_classes = os.listdir(gt_dirpath)

    for local_class in local_classes:
        class_dirpath = os.path.join(gt_dirpath, local_class)

        if os.path.isdir(class_dirpath):
            gt_train_annot_fpath = os.path.join(
                class_dirpath, gt_train_csv_fname)
            gt_test_annot_fpath = os.path.join(
                class_dirpath, gt_test_csv_fname)

            train_classes = pandas_get_unique_classes(gt_train_annot_fpath)
            test_classes = pandas_get_unique_classes(gt_test_annot_fpath)

            for _class in train_classes:
                class_name = local_class + '/' + _class

                if class_name not in unique_classes:
                    unique_classes += [class_name]

            gt_train_annot_fpaths += [(local_class, gt_train_annot_fpath)]
            gt_test_annot_fpaths += [(local_class, gt_test_annot_fpath)]

print(gt_train_annot_fpaths)
print(gt_test_annot_fpaths)
# print(unique_classes)

dst_data_dir = args.output

dst_train_data_dir = dst_data_dir + '_Train'
dst_test_data_dir = dst_data_dir + '_Test'

print('--- Copy frames to dst folder ---')

vc.copy_2_images_dir(dst_train_data_dir, gt_train_frame_fpaths)
vc.copy_2_images_dir(dst_test_data_dir, gt_test_frame_fpaths)

FILENAME_KEY = 'filename'
LEFT_X_KEY = 'x_from'
UUPPER_Y_KEY = 'y_from'
WIDTH_KEY = 'width'
HEIGHT_KEY = 'height'
SIGN_CLS_KEY = 'sign_class'

print('--- Reading annotations ---')


def read_annotations(list_annot_fpath):
    checked_files = {}
    for annot in tqdm.tqdm(list_annot_fpath):
        class_name, annot_fpath = annot

        df = pd.read_csv(annot_fpath)
        for index, row in df.iterrows():
            fname = row[FILENAME_KEY]
            xmin = int(row[LEFT_X_KEY])
            ymin = int(row[UUPPER_Y_KEY])
            xmax = int(xmin) + int(row[WIDTH_KEY])
            ymax = int(ymin) + int(row[HEIGHT_KEY])

            full_cls_name = class_name + '/' + row[SIGN_CLS_KEY]
            info = (xmin, ymin, xmax, ymax, full_cls_name)

            if fname in checked_files and info not in checked_files[fname]:
                checked_files[fname] += [info]
            else:
                checked_files[fname] = [info]

    return checked_files


def append_frames_wo_annot(checked_files, frames_fpath_list):
    for fpath in frames_fpath_list:
        fname = os.path.basename(fpath)
        if fname not in checked_files:
            checked_files[fname] = None


train_checked_files = read_annotations(gt_train_annot_fpaths)
test_checked_files = read_annotations(gt_test_annot_fpaths)

print('--- Append frames without annotation ---')

append_frames_wo_annot(train_checked_files, gt_train_frame_fpaths)
append_frames_wo_annot(test_checked_files, gt_test_frame_fpaths)

print('--- Creating annotations ---')

vc.generate_annotations(
    dst_train_data_dir, train_checked_files, args.show_frames)
vc.generate_annotations(
    dst_test_data_dir, test_checked_files, args.show_frames)
