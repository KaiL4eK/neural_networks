
import argparse
import json
import cv2
import numpy as np
import os

from time import time

import tensorflow as tf
import keras.backend as K
import pycuda.driver as cuda
import pycuda.autoinit

from frontend import YOLO

import tensorrt.parsers.uffparser as uffparser
import uff

import tensorrt as trt
from tensorflow.contrib import tensorrt as trt2

from utils import decode_netout

argparser = argparse.ArgumentParser(
    description='TensorRT based inference for network')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-p',
    '--pbpath',
    default='',
    help='path to PB (frozen TF graph) converted file')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-o',
    '--output',
    default='model',
    help='output frozen graph filename')

def _main_(args):

    config_path  = args.conf
    pb_path      = args.pbpath
    image_path   = args.input
    output_fname = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    # if weights_path == '':
        # weights_path = config['train']['saved_weights_name']

    ###############################
    #   Make the model 
    ###############################

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(sess)
    K.set_learning_phase(0)

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'],config['model']['input_size_w']), 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                trainable           = config['model']['trainable'],
                gray_mode           = config['model']['gray_mode'])

    ###############################
    #   Load trained weights
    ###############################    

    # yolo.load_weights(weights_path)

    model = yolo.model

    test_count = 100
 
    # Inference
    orig_image = cv2.imread(image_path)
    orig_image = cv2.resize(orig_image, (config['model']['input_size_w'], config['model']['input_size_h']))
    image = np.array(orig_image).reshape((config['model']['input_size_w'], config['model']['input_size_h'], 3))

    # Preprocess
    image = image.astype('float32')
    image = image / 127.5 - 1

    dummy_array = np.zeros((1,1,1,1,config['model']['max_box_per_image'],4))

    time_k1 = time()
    
    for i in range(test_count):
        keras_result = model.predict([[image], dummy_array])[0]
    
    time_k2 = time()


    # boxes  = decode_netout(netout, self.anchors, self.nb_class)



    # print(model.layers[-4].data_format)
    # exit(1)

    model_output = 'YOLO_output/Reshape'
    model_output = 'DetectionLayer/BiasAdd'
    # model_output = model.layers[-4].name

    # MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")
    # img, label = MNIST_DATASETS.test.next_batch(1)
    # img = img[0]
    # print(img.shape)

    # exit(1)

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, [model_output])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_fname
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    #     parser.register_input("input_1", (3, config['model']['input_size_w'], config['model']['input_size_h']))
    #     parser.register_output(model_output)
    #     parser.parse(pb_path, network)

    #     with builder.build_cuda_engine(network) as engine:


    #         exit(1)



    
    # stream = uff.from_tensorflow_frozen_model(pb_path, [model_output])

    # trt_graph_def = trt2.create_inference_graph(
    #                         frozen_graph,
    #                         [model_output],
    #                         max_batch_size=1,
    #                         max_workspace_size_bytes=(2 << 10) << 20,
    #                         precision_mode='FP32')


    uff_model = uff.from_tensorflow(frozen_graph, [model_output])
    
    log_sev = trt.infer.LogSeverity.ERROR
    G_LOGGER = trt.infer.ConsoleLogger(log_sev)

    trt_input_name = 'input_1'
    trt_input_shape = (3, config['model']['input_size_h'], config['model']['input_size_w'])

    trt_output_shape = (1, 1, 13, 13, 30)
    yolo_output_shape = (13, 13, 5, 6)

    data_type = 'FP16'

    image_chw = np.moveaxis(image, -1, 0)
    image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)

    if 1:       

        parser = uffparser.create_uff_parser()

        # kNCHW = 0
        # kNHWC = 1
        parser.register_input(trt_input_name, trt_input_shape, 0)
        parser.register_output(model_output)

        B2GB = 1/ (1024 * 1024 * 1024.0)
        print("Pre-engine memory: %.2f / %.2f" % (cuda.mem_get_info()[0] * B2GB, cuda.mem_get_info()[1] * B2GB))
        # exit(1)

        engine = trt.utils.uff_to_trt_engine(   logger=G_LOGGER, 
                                                stream=uff_model, 
                                                parser=parser, 
                                                max_batch_size=1, 
                                                max_workspace_size=(1 << 20), 
                                                datatype=data_type)

        print("Post-engine memory: %.2f / %.2f" % (cuda.mem_get_info()[0] * B2GB, cuda.mem_get_info()[1] * B2GB))

        parser.destroy()

        runtime = trt.infer.create_infer_runtime(G_LOGGER)
        context = engine.create_execution_context()
        
        trt_result1 = np.empty(yolo_output_shape, dtype = np.float32)

        d_input = cuda.mem_alloc(1 * image_chw.nbytes)
        d_output = cuda.mem_alloc(1 * trt_result1.nbytes)

        bindings = [int(d_input), int(d_output)]

        stream = cuda.Stream()

        time11 = time()

        for i in range(test_count):
            cuda.memcpy_htod_async(d_input, image_chw, stream)
            # Execute model
            context.enqueue(1, bindings, stream.handle, None)
            # Transfer predictions back
            cuda.memcpy_dtoh_async(trt_result1, d_output, stream)
            # Syncronize threads
            stream.synchronize()

        time12 = time()

        context.destroy()
        engine.destroy()
        # new_engine.destroy()
        runtime.destroy()

    if 1:
        engine = trt.lite.Engine(   log_sev=log_sev,
                                    framework='uff',
                                    stream=uff_model,
                                    input_nodes={trt_input_name: trt_input_shape},
                                    output_nodes=[model_output],
                                    max_batch_size=1,
                                    max_workspace_size=(1 << 20),
                                    datatype=data_type)

        time21 = time()

        for i in range(test_count):
            trt_result2 = engine.infer([image_chw])

        time22 = time()

    trt_result1   = np.array(trt_result1)
    trt_result2   = np.array(trt_result2)

    print(trt_result1.shape, trt_result2.shape)

    print('TRT1 %.2f / TRT2 %.2f / Keras %.2f' % (time12-time11, time22-time21, time_k2-time_k1))

    trt_result1 = trt_result1.reshape(yolo_output_shape)
    trt_result2 = trt_result2.reshape(yolo_output_shape)

    print(trt_result1.shape, trt_result2.shape, keras_result.shape)
    print(trt_result1[0, 0, 0])
    print(trt_result2[0, 0, 0])
    print(keras_result[0, 0, 0])


    # print(np.array_equal(trt_result1, trt_result2))

    # trt_result   = np.array(trt_result)
    # keras_result = np.array(keras_result)
    # trt_result = trt_result.reshape((13, 13, 5, 6))

    # print(trt_result[0, 0, 0], keras_result[0, 0, 0])

    # print(trt_result.shape, keras_result.shape)
    # print(np.array_equal(trt_result, keras_result))

        # print(result.shape)




if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
