import mvnc.mvncapi as fx
import numpy as np
import cv2


class InferNCS:
    def __init__(self, graph_fpath, device_idx=0, fp16=True):
        # fx.global_set_option(fx.GlobalOption.RW_LOG_LEVEL, 0)
        self.dev = None
        self.graph = None
        self.fifoIn = None
        self.fifoOut = None

        devices = fx.enumerate_devices()
        if len(devices) < 1:
            print("Error - no NCS devices detected, verify an NCS device is connected.")
            return

        if device_idx > len(devices):
            print("Error - device NCS idx is out of range.")
            return

        self.fp16 = fp16
        self.dev = fx.Device(devices[device_idx])

        try:
            self.dev.open()
        except:
            print("Error - Could not open NCS device.")
            quit()

        print("Hello NCS! Device opened normally.")

        print("Opening graph: {}".format(graph_fpath))
        with open(graph_fpath, mode='rb') as f:
            graph_file_buff = f.read()

        self.graph = fx.Graph('graph')

        print("FIFO Allocation / FP16: {}".format(fp16))

        if self.fp16:
            self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(self.dev, graph_file_buff,
                                                                       input_fifo_data_type=fx.FifoDataType.FP16,
                                                                       output_fifo_data_type=fx.FifoDataType.FP16)
        else:
            self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(self.dev, graph_file_buff)

        output_tensor_list = self.graph.get_option(fx.GraphOption.RO_OUTPUT_TENSOR_DESCRIPTORS)
        self.output_shape = (output_tensor_list[0].h, output_tensor_list[0].w, output_tensor_list[0].c)

        input_tensor_list = self.graph.get_option(fx.GraphOption.RO_INPUT_TENSOR_DESCRIPTORS)
        self.input_shape = (input_tensor_list[0].h, input_tensor_list[0].w, input_tensor_list[0].c)
        self.input_cv_sz = (input_tensor_list[0].w, input_tensor_list[0].h)

        print('Input shape: {}'.format(self.input_shape))
        print('Output shape: {}'.format(self.output_shape))

    def __del__(self):
        if self.fifoIn:
            self.fifoIn.destroy()

        if self.fifoOut:
            self.fifoOut.destroy()

        if self.graph:
            self.graph.destroy()

        if self.dev:
            self.dev.close()

        print("Close NCS")

    def is_opened(self):
        if self.dev is None:
            return False
        else:
            return True

    def infer(self, img):
        img_h, img_w, img_c = img.shape

        if img_c != self.input_shape[2]:
            print('Invalid number of channels')
            return None

        if img_w != self.input_cv_sz[0] or img_h != self.input_cv_sz[1]:
            img = cv2.resize(img, self.input_cv_sz)

        if self.fp16:
            img = img.astype(np.float16)
        else:
            img = img.astype(np.float32)

        self.graph.queue_inference_with_fifo_elem(self.fifoIn, self.fifoOut, img, None)

        try:
            ncs_output, _ = self.fifoOut.read_elem()
        except:
            print('Failed to read FIFO')
            ncs_output = np.zeros(self.output_shape)

        ncs_output = ncs_output.reshape(self.output_shape)

        return ncs_output

    def predict(self, img_batch):
        return np.expand_dims(self.infer(img_batch[0]), axis=0)

    def predict_on_batch(self, img_batch):
        return np.expand_dims(self.infer(img_batch[0]), axis=0)
