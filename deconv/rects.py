#!/usr/bin/python3
import cv2
import numpy as np


def get_rects(img, threshold):
    """Applies threshold operation and returns list of
    contour bounding rectangles in a form of a lists [x, y, width, height]

    img       - single channel 8-bit image
    threshold - threshold value"""
    threshold_result = np.zeros(img.shape, np.uint8)
    cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY, threshold_result)
    im2, contours, hierarchy = cv2.findContours(threshold_result,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
    rects = [list(cv2.boundingRect(contour)) for contour in contours]
    return rects


def draw_rects(img, rects, color=(255, 0, 0)):
    """Draws rectangles on a image

    img   - 3 channel BGR 8-bit image
    rects - list of rectangles [x, y, width, height]
    color - BGR color"""
    for rect in rects:
        cv2.rectangle(img, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]), color, 3)


def test():
    img = np.zeros((480, 640, 1), np.uint8)
    cv2.rectangle(img, (40, 60), (100, 100), 255, -1)
    cv2.rectangle(img, (140, 160), (400, 300), 255, -1)
    cv2.GaussianBlur(img, (55, 55), 0, dst=img)
    cv2.imshow("img", img)
    rects = get_rects(img, 128)
    print(rects)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_rects(color_img, rects)
    cv2.imshow("Rects", color_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
