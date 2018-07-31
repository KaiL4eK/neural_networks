#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("/usr/local/python")

import time 
import csv
import cv2
import numpy as np
import caffe
import gc
import os
import lmdb
import random
from PIL import Image


IMAGE_INPUT_WIDTH  = 240
IMAGE_INPUT_HEIGHT = 240

IMAGE_OUTPUT_WIDTH  = 240
IMAGE_OUTPUT_HEIGHT = 240

DEBUG = False

DATABASE_ROOT = '../Trafic_road_Training_db/Initial_db'

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "[{0}, {1}]".format(self.x, self.y)

class Rectangle(object):
    def __init__(self, ul=None, lr=None):
        self.ul = ul
        self.lr = lr

    def __str__(self):
        return "[{0}, {1}]".format(self.ul, self.lr)

class TrainData(object):
    def __init__(self, fname=None, rect=None):
        self.fname      = fname
        self.rects      = []

        if rect != None:
            self.rects.append(rect)

    def __str__(self):
        return "[Name: {0}, Rects: {1}]".format(self.fname, self.rects)

''' Source specific data '''

DATABASE_ROOT_2 = DATABASE_ROOT + '/2'

UPPER_LEFT_STR = 'Upper left corner '
LOWER_RIGHT_STR = 'Lower right corner '

def readTrainData_2():
    with open(DATABASE_ROOT_2 + '/allAnnotations_1.csv', 'rb') as annot_file:
        d_reader = csv.DictReader(annot_file)

        train_entries   = []

        for line in d_reader:
            upper_left  = Point(float(line[UPPER_LEFT_STR + 'X']), float(line[UPPER_LEFT_STR + 'Y']))
            lower_right = Point(float(line[LOWER_RIGHT_STR + 'X']), float(line[LOWER_RIGHT_STR + 'Y']))     

            fname = DATABASE_ROOT_2 + '/' + line['Filename']
            # img = cv2.imread(fname, cv2.IMREAD_COLOR)

            rect = Rectangle(upper_left, lower_right)

            tdata = filter(lambda x: x.fname == fname, train_entries)
            if train_entries and tdata:
                tdata[0].rects.append(rect)
            else:
                train_entries.append(TrainData(fname, rect))

    print len(train_entries)

    return train_entries

DATABASE_ROOT_5 = DATABASE_ROOT + '/5/FullIJCNN2013'

def readTrainData_5():
    train_entries   = []

    with open(DATABASE_ROOT_5 + '/gt.txt', 'rb') as annot_file:
        for row in annot_file:
            data = row.split(';')

            fname = DATABASE_ROOT_5 + '/' + data[0]
            upper_left  = Point(float(data[1]), float(data[2]))
            lower_right = Point(float(data[3]), float(data[4]))
            rect = Rectangle(upper_left, lower_right)

            tdata = filter(lambda x: x.fname == fname, train_entries)
            if train_entries and tdata:
                tdata[0].rects.append(rect)
            else:
                train_entries.append(TrainData(fname, rect))
    
    print len(train_entries)

    return train_entries

DATABASE_ROOT_1 = DATABASE_ROOT + '/1'

def readTrainData_1():
    train_entries   = []

    with open(DATABASE_ROOT_1 + '/annotations_s1.txt', 'rb') as annot_file:
        for row in annot_file:
            fname, info = row.split(':')

            fname = DATABASE_ROOT_1 + '/p1/' + fname

            # Unknown signs
            infos = info.split(';')
            if infos[0] == 'MISC_SIGNS':
                continue

            # Empty pictures
            if len(infos) == 1:
                train_entries.append(TrainData(fname))
                continue

            for info in infos:
                # Let`s get just visible pictures =)
                if info.startswith(('VISIBLE', 'BLURRED', 'OCCLUDED')):
                    info_bb = info.split(',');

                    upper_left  = Point(float(info_bb[3]), float(info_bb[4]))
                    lower_right = Point(float(info_bb[1]), float(info_bb[2]))
                    rect = Rectangle(upper_left, lower_right)

                    tdata = filter(lambda x: x.fname == fname, train_entries)
                    if train_entries and tdata:
                        tdata[0].rects.append(rect)
                    else:
                        train_entries.append(TrainData(fname, rect))

    with open(DATABASE_ROOT_1 + '/annotations_s2.txt', 'rb') as annot_file:
        for row in annot_file:
            fname, info = row.split(':')

            fname = DATABASE_ROOT_1 + '/p2/' + fname

            # Unknown signs
            infos = info.split(';')
            if infos[0] == 'MISC_SIGNS':
                continue

            # Empty pictures
            if len(infos) == 1:
                train_entries.append(TrainData(fname))
                continue

            for info in infos:
                # Let`s get just visible pictures =)
                if info.startswith(('VISIBLE', 'BLURRED', 'OCCLUDED')):
                    info_bb = info.split(',');

                    upper_left  = Point(float(info_bb[3]), float(info_bb[4]))
                    lower_right = Point(float(info_bb[1]), float(info_bb[2]))
                    rect = Rectangle(upper_left, lower_right)

                    tdata = filter(lambda x: x.fname == fname, train_entries)
                    if train_entries and tdata:
                        tdata[0].rects.append(rect)
                    else:
                        train_entries.append(TrainData(fname, rect))

    print len(train_entries)

    return train_entries


KOSTYA_DATABASE_PATH = DATABASE_ROOT + '/kostya_signs'

def readTrainData_Kostya():
    train_entries   = []

    for filepath in os.listdir(KOSTYA_DATABASE_PATH):
        info = filepath.split(';')
        ul_x = max(0, int(info[1]))
        ul_y = max(0, int(info[2]))
        lr_x = ul_x + max(0, int(info[3]))
        lr_y = ul_y + max(0, int(info[4]))
        
        fname = KOSTYA_DATABASE_PATH + '/' + filepath
        upper_left  = Point(float(ul_x), float(ul_y))
        lower_right = Point(float(lr_x), float(lr_y))
        rect = Rectangle(upper_left, lower_right)

        train_entries.append(TrainData(fname, rect))

    print len(train_entries)

    return train_entries


''' Common preparations '''

def generateOutputImage(dataEntry, img_orig, img_width=IMAGE_OUTPUT_WIDTH, img_height=IMAGE_OUTPUT_HEIGHT):
    
    height, width, channels = img_orig.shape

    scale_x = img_width * 1. / width
    scale_y = img_height * 1. / height

    blank_image = np.ones((img_height,img_width,1), np.uint8)

    for rect in dataEntry.rects:
        ul_x = int(rect.ul.x * scale_x)
        ul_y = int(rect.ul.y * scale_y)
        lr_x = int(rect.lr.x * scale_x)
        lr_y = int(rect.lr.y * scale_y)

        # print rect, ul_x, ul_y

        cv2.rectangle(blank_image, (ul_x, ul_y), (lr_x, lr_y), thickness=-1, color=0 )

    if DEBUG:
        cv2.imshow('2', np.uint8(blank_image) * 255)

    blank_image = np.reshape(blank_image, (img_height, img_width, 1))
    blank_image = blank_image.transpose((2,0,1))

    # print np.unique(blank_image)

    return np.uint8(blank_image)

def prepareInputImage(img_orig, img_width=IMAGE_INPUT_WIDTH, img_height=IMAGE_INPUT_HEIGHT):

    img = img_orig

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_LINEAR)
    
    if DEBUG:
        cv2.imshow('1', img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.transpose((2,0,1))

    return np.uint8(img)

def create_img_lmdb(lmdb_fname, data_entries):

    in_db = lmdb.open('data/input_' + lmdb_fname, map_size=int(1e12))
    out_db = lmdb.open('data/output_' + lmdb_fname, map_size=int(1e12))

    skipped = 0

    with in_db.begin(write=True) as in_txn, out_db.begin(write=True) as out_txn:
        for idx, entry in enumerate(data_entries):

            # Try to search one signs
            if len(entry.rects) <= 1:
                # load image:
                # - as np.uint8 {0, ..., 255}
                # - in BGR (switch from RGB)
                # - in Channel x Height x Width order (switch from H x W x C)
                img_orig = cv2.imread(entry.fname, cv2.IMREAD_COLOR)

                out_im = generateOutputImage(entry, img_orig)
                in_im = prepareInputImage(img_orig)

                sum_of_el = 57600 - np.sum(out_im)
                if sum_of_el < 100 and sum_of_el > 0:
                    skipped += 1
                    continue

                if DEBUG:
                    # print out_im.shape, in_im.shape
                    if cv2.waitKey(0) == 27:
                        break

                im_dat = caffe.io.array_to_datum(in_im)
                in_txn.put('{:0>10d}'.format(idx-skipped), im_dat.SerializeToString())

                im_dat = caffe.io.array_to_datum(out_im)
                out_txn.put('{:0>10d}'.format(idx-skipped), im_dat.SerializeToString())
            else:
                skipped += 1

            if idx % 100 == 0:
                print 'Ready:', idx, 'skipped:', skipped

            # if idx >= 1:
                # break

    in_db.close()
    out_db.close()

    print 'Done:', (len(data_entries) - skipped), 'Skiped:', skipped



if __name__ == '__main__':
    tdata = []
# Monochrome
#    tdata = tdata + readTrainData_2()

    # tdata = tdata + readTrainData_1()
    # tdata = tdata + readTrainData_5()

    tdata = tdata + readTrainData_Kostya()

    random.shuffle(tdata)

    print 'Full size:', len(tdata)

    vdata = tdata[::11]
    del tdata[::11]

    print 'Division', len(tdata), '/', len(vdata)

    create_img_lmdb('train_lmdb', tdata)
    create_img_lmdb('verify_lmdb', vdata)
