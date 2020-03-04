
import os
from shutil import copyfile


from pathlib import Path

ANNOT_DIR = '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF19/Annotations/'
IMAGES_SRC_DIR = '/home/alexey/data/NN_data/robofest_data/__RF19/Images.full/'
IMAGES_DST_DIR = '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF19/Images/'

annot_fstems = [Path(fname).stem for fname in os.listdir(ANNOT_DIR) if fname.endswith('.xml')]
images_fnames = [fname for fname in os.listdir(IMAGES_SRC_DIR)]

for ann_fstem in annot_fstems:
    image_fname = ann_fstem + '.png'
    if image_fname in images_fnames:
        src_fpath = os.path.join(IMAGES_SRC_DIR, image_fname)
        dst_fpath = os.path.join(IMAGES_DST_DIR, image_fname)
        copyfile(src_fpath, dst_fpath)

    image_fname = ann_fstem + '.jpg'
    if image_fname in images_fnames:
        src_fpath = os.path.join(IMAGES_SRC_DIR, image_fname)
        dst_fpath = os.path.join(IMAGES_DST_DIR, image_fname)
        copyfile(src_fpath, dst_fpath)

    image_fname = ann_fstem + '.jpeg'
    if image_fname in images_fnames:
        src_fpath = os.path.join(IMAGES_SRC_DIR, image_fname)
        dst_fpath = os.path.join(IMAGES_DST_DIR, image_fname)
        copyfile(src_fpath, dst_fpath)


