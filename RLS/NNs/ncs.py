import mvnc.mvncapi as fx
import numpy as np
from utils import sigmoid

class InferNCS:
    def __init__(self, graph_fpath, fp16=True):
        # fx.global_set_option(fx.GlobalOption.RW_LOG_LEVEL, 0)

        devices = fx.enumerate_devices()
        if (len(devices) < 1):
            print("Error - no NCS devices detected, verify an NCS device is connected.")
            quit() 

        self.fp16 = fp16
        self.dev = fx.Device(devices[0])
        
        try:
            self.dev.open()
        except:
            print("Error - Could not open NCS device.")
            quit()

        print("Hello NCS! Device opened normally.")
        
        with open(graph_fpath, mode='rb') as f:
            graphFileBuff = f.read()

        self.graph = fx.Graph('graph')



        print("FIFO Allocation")

        if self.fp16:
            self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(self.dev, graphFileBuff, 
                                            input_fifo_data_type=fx.FifoDataType.FP16, 
                                            output_fifo_data_type=fx.FifoDataType.FP16)
        else:
            self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(self.dev, graphFileBuff)

        output_tensor_list = self.graph.get_option(fx.GraphOption.RO_OUTPUT_TENSOR_DESCRIPTORS)
        self.output_shape = (output_tensor_list[0].h, output_tensor_list[0].w, output_tensor_list[0].c)

    def __del__(self):
        self.fifoIn.destroy()
        self.fifoOut.destroy()
        self.graph.destroy()
        self.dev.close()


    def infer(self, img):

        if self.fp16:
            img = img.astype(np.float16)
        else:
            img = img.astype(np.float32)

        self.graph.queue_inference_with_fifo_elem(self.fifoIn, self.fifoOut, img, None)

        ncs_output, _ = self.fifoOut.read_elem()

        ncs_output = sigmoid(ncs_output)
        ncs_output = ncs_output.reshape(self.output_shape)

        return ncs_output
