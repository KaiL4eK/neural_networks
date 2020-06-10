
import os
from pathlib import Path

ANNOT_DIRS = [
    '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF17/Annotations/',
    '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF19/Annotations/',
]

IMAGES_DIRS = [
    '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF17/Images/',
    '/home/alexey/data/NN_experiments/TSD/data/RF/TSD/RF19/Images/'
]

annot_fstems = []
for ann_dir in ANNOT_DIRS:
    for fname in os.listdir(ann_dir):
        annot_fstems.append(
            Path(fname).stem
        )

images_fstems = []
for img_dir in IMAGES_DIRS:
    for fname in os.listdir(img_dir):
        images_fstems.append(
            Path(fname).stem
        )

imgs_wo_annots = set(images_fstems) - (set(images_fstems) & set(annot_fstems))
anns_wo_images = set(annot_fstems) - (set(images_fstems) & set(annot_fstems))

print(imgs_wo_annots)
print(anns_wo_images)

for ann in anns_wo_images:
    for ann_dir in ANNOT_DIRS:
        try:
            pass
            os.remove(os.path.join(ann_dir, ann+'.xml'))
        except:
            pass