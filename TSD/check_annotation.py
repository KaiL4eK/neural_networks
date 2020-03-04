
import os
from pathlib import Path

ANNOT_DIR = '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF19/Annotations/'
IMAGES_DIR = '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF19/Images/'

annot_fstems = [Path(fname).stem for fname in os.listdir(ANNOT_DIR) if fname.endswith('.xml')]
images_fstems = [Path(fname).stem for fname in os.listdir(IMAGES_DIR)]

imgs_wo_annots = set(images_fstems) - (set(images_fstems) & set(annot_fstems))
anns_wo_images = set(annot_fstems) - (set(images_fstems) & set(annot_fstems))

print(imgs_wo_annots)
print(anns_wo_images)
