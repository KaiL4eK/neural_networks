import sys
sys.path.insert(0, 'TensorRT/src/')
import tensorNet

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    GLOBAL_TRT_IMPORT_FAIL = False
except:
    GLOBAL_TRT_IMPORT_FAIL = True
    print('TensorRT import failed, enable Cpp variant')

import numpy as np
import os

import json

from utils.utils import correct_yolo_boxes, do_nms, decode_netout


class TRTengine:
    def __init__ (self, isCppInf=False):

        self.isCppInf = isCppInf

        if GLOBAL_TRT_IMPORT_FAIL:
            self.isCppInf = True

        if not self.isCppInf:
            log_sev = trt.infer.LogSeverity.ERROR
            self.G_LOGGER = trt.infer.ConsoleLogger(log_sev)

        self.engine_cfg = {}
        self.engine_cfg['trt'] = {}

        self.engine_cfg['trt']['data_type'] = 'FP32'

        self.engine = None

    def import_from_weights(self, config, weights):

        import tensorflow as tf
        import keras.backend as K
        from yolo import create_model

        import uff
        
        import tensorrt.parsers.uffparser as uffparser

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        K.set_session(sess)
        K.set_learning_phase(0)

        self.config = config

        self.engine_fname = 'engines/trtEngine_{}.trt'.format(config['model']['base'])
        self.engine_cfg_fname = 'engines/trtEngine_{}.cfg'.format(config['model']['base'])
        self.engine_uff_fname = 'engines/trtEngine_{}.uff'.format(config['model']['base'])

        self.engine_cfg['labels'] = self.config['model']['labels']

        self.nb_classes = len(self.config['model']['labels'])
        self.engine_cfg['anchors'] = self.config['model']['anchors']

        self.input_h, self.input_w = self.config['infer']['input_sz'], self.config['infer']['input_sz']

        train_model, infer_model, freeze_num = create_model(
            nb_class            = self.nb_classes, 
            anchors             = self.engine_cfg['anchors'], 
            max_box_per_image   = 0, 
            max_input_size      = self.config['model']['max_input_size'], 
            batch_size          = self.config['train']['batch_size'], 
            base                = self.config['model']['base'],
            img_shape           = (self.input_h, self.input_w, 3),
            load_src_weights    = False
        )

        infer_model.load_weights( weights, by_name=True, skip_mismatch=True ) 

        self.keras_model = infer_model

        self.engine_cfg['n_outputs'] = len(self.keras_model.outputs)

        self.n_anchors = len(self.engine_cfg['anchors']) // 2 // self.engine_cfg['n_outputs']

#################

        self.engine_cfg['trt']['trt_output_shapes'] = [(int(output.shape[1]), int(output.shape[2]), self.n_anchors, 4+1+self.nb_classes) for output in self.keras_model.outputs]
        self.engine_cfg['trt']['trt_input_shape'] = (3, self.input_h, self.input_w)

#################

        self.model_outputs = [output.name.split(':')[0] for output in self.keras_model.outputs]
        self.model_input = self.keras_model.inputs[0].name.split(':')[0]

        graphdef = sess.graph.as_graph_def()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, self.model_outputs)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

        uff_model = uff.from_tensorflow(frozen_graph, self.model_outputs, output_filename=self.engine_uff_fname)

        parser = uffparser.create_uff_parser()

        # kNCHW = 0
        # kNHWC = 1
        parser.register_input(self.model_input, self.engine_cfg['trt']['trt_input_shape'], 1)
        for output in self.model_outputs:
            parser.register_output(output)

        B2GB = 1/ (1024 * 1024 * 1024.0)
        print("Pre-engine memory: %.2f / %.2f" % (cuda.mem_get_info()[0] * B2GB, cuda.mem_get_info()[1] * B2GB))
        # exit(1)

        self.engine = trt.utils.uff_to_trt_engine(
                            logger=self.G_LOGGER, 
                            stream=uff_model, 
                            parser=parser, 
                            max_batch_size=1, 
                            max_workspace_size=(1 << 25), 
                            datatype=self.engine_cfg['trt']['data_type']
                          )

        print("Post-engine memory: %.2f / %.2f" % (cuda.mem_get_info()[0] * B2GB, cuda.mem_get_info()[1] * B2GB))

        parser.destroy()

        self.init_engine()

    # https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt_401/tensorrt-api/python_api/pkg_ref/lite.html#tensorrt.lite.Engine

    def init_engine(self):
    
        if not self.engine:
            return

        self.trt_results = [np.empty(shp, dtype = np.float32) for shp in self.engine_cfg['trt']['trt_output_shapes']]

        self.input_h = self.engine_cfg['trt']['trt_input_shape'][1]        
        self.input_w = self.engine_cfg['trt']['trt_input_shape'][2]        
        self.anchors_grp = [self.engine_cfg['anchors'][(self.engine_cfg['n_outputs']-1-j)*6:(self.engine_cfg['n_outputs']-j)*6] for j in range(self.engine_cfg['n_outputs'])]

        if not self.isCppInf:
            self.runtime = trt.infer.create_infer_runtime(self.G_LOGGER)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.trt_d_outputs = [cuda.mem_alloc(1 * result.nbytes) for result in self.trt_results]

            self.trt_input_dummy = np.empty(self.engine_cfg['trt']['trt_input_shape'], dtype = np.float32)
            self.trt_d_input = cuda.mem_alloc(1 * self.trt_input_dummy.nbytes)

            self.trt_bindings = [int(self.trt_d_input)] + [int(out) for out in self.trt_d_outputs]

    def get_labels(self):
        return self.engine_cfg['labels']

    def predict_boxes(self, image, obj_thresh=.5, nms_thresh=.45):

        image_h, image_w, _ = image.shape

        input_img = preprocess_input(image, self.input_h, self.input_w)

        # Convert 2 CHW
        image_chw = np.moveaxis(input_img, -1, 0)
        image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)

        boxes = []

        if self.isCppInf:

            tensorNet.inference(self.engine, image_chw)

            for j in range(self.engine_cfg['n_outputs']):
                tensorNet.getOutput(self.engine, j, self.trt_results[j])
                boxes += decode_netout(self.trt_results[j], self.anchors_grp[j], obj_thresh, self.input_h, self.input_w)

        else:

            # Predict
            cuda.memcpy_htod_async(self.trt_d_input, image_chw, self.stream)
            # Execute model
            self.context.enqueue(1, self.trt_bindings, self.stream.handle, None)
            # Transfer predictions back

            for i in range(self.engine_cfg['n_outputs']):
                cuda.memcpy_dtoh_async(self.trt_results[i], self.trt_d_outputs[i], self.stream)

            # Syncronize threads
            self.stream.synchronize()

            # decode the output of the network
            for j in range(self.engine_cfg['n_outputs']):
                boxes += decode_netout(self.trt_results[j], self.anchors_grp[j], obj_thresh, self.input_h, self.input_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, self.input_h, self.input_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)        

        return boxes



    def save_engine(self):

        if not self.engine:
            return

        if not os.path.exists('engines'):
            os.makedirs('engines')

        trt.utils.write_engine_to_file(self.engine_fname, self.engine.serialize())

        with open(self.engine_cfg_fname, 'w') as outfile:
            json.dump(self.engine_cfg, outfile, sort_keys = True, indent = 4)


    def load_engine(self, engine_fname):

        self.engine_fname       = engine_fname
        self.engine_cfg_fname   = engine_fname.split('.')[0] + '.cfg'

        if not os.path.exists(self.engine_fname):
            return False

        if not os.path.exists(self.engine_cfg_fname):
            return False

        if self.isCppInf:
            # import pdb
            
            if self.engine_fname.endswith('uff'):
                self.engine = tensorNet.createTrtFromUFF(self.engine_fname)
                tensorNet.saveEngine(self.engine, self.engine_fname.split('.')[0] + '.trt')
            else:
                self.engine = tensorNet.createTrtFromPlan(self.engine_fname)
            
            print('Created TensorRT engine from {} in Cpp infer'.format(self.engine_fname))

            # pdb.set_trace()

            tensorNet.showEngineSummary(self.engine)
            # tensorNet.prepareBuffer(self.engine)
            
        else:
            
            self.engine = trt.utils.load_engine(self.G_LOGGER, self.engine_fname)
            print('Created TensorRT engine from {} in Py infer'.format(self.engine_fname))

        with open(self.engine_cfg_fname) as infile:
            self.engine_cfg = json.load(infile)

        self.init_engine()

        return True
