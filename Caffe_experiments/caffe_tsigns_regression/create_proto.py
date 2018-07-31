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
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, 
                                 weight_filler=dict(type='gaussian', std=0.0001), bias_filler=dict(type='constant'))
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
    n.conv1, n.relu1 = conv_relu(n.data, ks=5, nout=20, stride=1)
    n.pool1 = max_pool(n.relu1, ks=2, stride=2)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.1)
    n.conv2, n.relu2 = conv_relu(n.drop1, ks=5, nout=48, stride=1)
    n.pool2 = max_pool(n.relu2, ks=2, stride=2)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.3)
    n.conv3, n.relu3 = conv_relu(n.drop2, ks=3, nout=64, stride=1)
    n.drop3 = L.Dropout(n.relu3, dropout_ratio=0.5)
    n.fc5 = L.InnerProduct(n.drop3, inner_product_param=dict(num_output=500, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
    n.drop4 = L.Dropout(n.fc5, dropout_ratio=0.5)
    n.fc6 = L.InnerProduct(n.drop4, inner_product_param=dict(num_output=4, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')))
    
    n.loss = L.EuclideanLoss(n.fc6, n.label, loss_weight=1)

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

TEST_SAVE_INTERVAL = int(1000)

def generate_solver():
    solver_prototxt = tools.solvers.SolverProtoTXT({
        'train_net': TRAIN_NET_FILE,
        # Used for SGD stabilization
        'iter_size': 1,
        'test_net': TEST_NET_FILE,
        'test_initialization': 'false',
        'test_iter': 31,
        'test_interval': TEST_SAVE_INTERVAL,
        'base_lr': 1e-5,
        'lr_policy': 'poly',
        #'gamma': 0.0001,
        'power': 3,
        # 'stepsize': 1000,
        'display': 50,
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
    make_net()
    generate_solver()
