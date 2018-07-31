from __future__ import print_function

import sys
sys.path.append("/usr/local/python")
sys.path.append("../caffe-libs/caffe-tools")

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from caffe.coord_map import crop
import caffe
import math as m
import caffe
import tools.solvers

DATA_ROOT = 'data'

TRAIN_LMDB_DATA_FILE = DATA_ROOT + '/input_train_lmdb'
TRAIN_LMDB_LABEL_FILE = DATA_ROOT + '/output_train_lmdb'
TRAIN_MEAN_FILE = DATA_ROOT + '/input_train_mean.binaryproto'

TEST_LMDB_DATA_FILE = DATA_ROOT + '/input_verify_lmdb'
TEST_LMDB_LABEL_FILE = DATA_ROOT + '/output_verify_lmdb'
TEST_MEAN_FILE = DATA_ROOT + '/input_verify_mean.binaryproto'

SOLVER_FILE = 'solver.prototxt'
TRAIN_NET_FILE = 'train.prototxt'
TEST_NET_FILE = 'test.prototxt'

# TRAIN_SET_SIZE = 6690
# TEST_SET_SIZE = 669
MAX_ITERATIONS = int(1e7)

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# mean_value=104.00698793, mean_value=116.66876762, mean_value=122.67891434

# weight_filler=dict(type='bilinear')
# weight_filler=dict(type='gaussian', std=0.0001), bias_filler=dict(type='constant')
def generate_net():
    n = caffe.NetSpec()

    n.data = L.Data(source=TRAIN_LMDB_DATA_FILE, backend=P.Data.LMDB, batch_size=1, ntop=1, 
                    transform_param=dict(scale=1./255))
    n.label = L.Data(source=TRAIN_LMDB_LABEL_FILE, backend=P.Data.LMDB, batch_size=1, ntop=1)

    # the base net
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, pad=100)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 6, 4096)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 1, 4096)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    # weight_filler=dict(type='gaussian', std=0.0001), bias_filler=dict(type='constant')
    n.score_fr_ = L.Convolution(n.drop7, num_output=2, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.upscore_ = L.Deconvolution(n.score_fr_,
        convolution_param=dict(num_output=2, kernel_size=63, stride=32, group=2,
            bias_term=False, weight_filler=dict(type='bilinear')),
        param=[dict(lr_mult=0)])

    n.score = crop(n.upscore_, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_weight=1,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()

'''
From https://github.com/BVLC/caffe/blob/master/src/caffe/solvers/sgd_solver.cpp
- fixed: always return base_lr.
- step: return base_lr * gamma ^ (floor(iter / step))
- exp: return base_lr * gamma ^ iter
- inv: return base_lr * (1 + gamma * iter) ^ (- power)
- multistep: similar to step but it allows non uniform steps defined by stepvalue
- poly: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
- sigmoid: the effective learning rate follows a sigmod decay return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
'''

TEST_SAVE_INTERVAL = int(50)

def generate_solver():
    solver_prototxt = tools.solvers.SolverProtoTXT({
        'train_net': TRAIN_NET_FILE,
        # Used for SGD stabilization
        'iter_size': 256,
        'test_net': TEST_NET_FILE,
        'test_initialization': 'false',
        'test_iter': 70,
        'test_interval': TEST_SAVE_INTERVAL,
        'base_lr': 1e-6,
        'lr_policy': 'poly',
        #'gamma': 0.0001,
        'power': 3,
        # 'stepsize': 1000,
        'display': 1,
        'max_iter': MAX_ITERATIONS,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'snapshot': TEST_SAVE_INTERVAL,
        #'solver_mode': 'GPU',
        'snapshot_prefix': 'snapshot/snapshot'        
    })
        
    solver_prototxt.write(SOLVER_FILE)

def make_net():
    with open(TRAIN_NET_FILE, 'w') as f:
        print(generate_net(), file=f)

    with open(TEST_NET_FILE, 'w') as f:
        print(generate_net(), file=f)

if __name__ == '__main__':
    batch = 1

    make_net()
    generate_solver()
