import cv2
import numpy as np


# def resize2rect(img, rect_hw, fill):
#     new_h, new_w, _ = image.shape
#     net_h, net_w = input_hw

#     # determine the new size of the image
#     if (float(net_w) / new_w) < (float(net_h) / new_h):
#         new_h = (new_h * net_w) // new_w
#         new_w = net_w
#     else:
#         new_w = (new_w * net_h) // new_h
#         new_h = net_h

#     resized = cv2.resize(image, (new_w, new_h))
#     resized = normalize(resized)

#     # embed the image into the standard letter box
#     new_image = np.zeros((net_h, net_w, 3))
#     new_image[(net_h-new_h)//2:(net_h+new_h)//2,
#               (net_w-new_w)//2:(net_w+new_w)//2, :] = resized

#     return new_image
