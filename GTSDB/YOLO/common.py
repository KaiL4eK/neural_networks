import keras.backend as K
import numpy as np

CHW_FORMAT = 'channels_first'
HWC_FORMAT = 'channels_last'

def getDataFormat():
    return K.image_data_format()

def setCHWDataFormat():
    K.set_image_data_format(CHW_FORMAT)

def setHWCDataFormat():
    K.set_image_data_format(HWC_FORMAT)

def isDataFormatCv():
    return (K.image_data_format() == HWC_FORMAT)

def convertHWC2CHW(img):
    image_chw = np.moveaxis(img, -1, 0)
    image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)
    return image_chw

