import cv2

def image_preprocess(image, net_h, net_w):
    resized = cv2.resize(image, (net_w, net_h))

    return normalize(resized)

def normalize(image):
    return image/255.
