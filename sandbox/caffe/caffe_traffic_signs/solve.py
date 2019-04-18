import sys
sys.path.append('/usr/local/python')

import caffe

import numpy as np
import os
import cv2

# try:
#     import setproctitle
#     setproctitle.setproctitle(os.path.basename(os.getcwd()))
# except:
#     pass

weights = 'weights'
# weights = 'fcn-alexnet-pascal.caffemodel'

# init
# caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
# interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
# surgery.interp(solver.net, interp_layers)

# scoring
# val = np.loadtxt('segvalid11.txt', dtype=str)
print 'Starts learning'

for _ in range(100000):
    solver.step(1)

    net = solver.net
    img_in = net.blobs['data'].data[0,:] * 255
    img_in = img_in.transpose((1,2,0))
    img_in = np.uint8(img_in)
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

    img_out = net.blobs['label'].data[0,0] * 255
    img_out = np.uint8(img_out)

    img_res = net.blobs['score'].data[0,0]
    img_res = np.uint8(img_res)

    img_res1 = net.blobs['score'].data[0,1]
    img_res1 = np.uint8(img_res1)

    # print( net.params['upscore_'][0].data[...] )

    # img_c1 = net.blobs['Convolution1'].data[0,0,:] * 255
    # img_c1 = img_c1.transpose()
    # img_c1 = np.uint8(img_c1)

    # print(net.params['Convolution1'][0].data)

    cv2.imshow('out', img_out)
    cv2.imshow('in', img_in)
    cv2.imshow('res', img_res)
    # cv2.imshow('res1', img_res1)
    # cv2.imshow('c1', img_c1)

    cv2.waitKey(0)
    # score.seg_tests(solver, False, val, layer='score')
