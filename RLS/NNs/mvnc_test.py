import mvnc.mvncapi as fx

# main entry point for the program
if __name__=="__main__":

     # set the logging level for the NC API
    fx.global_set_option(fx.GlobalOption.RW_LOG_LEVEL, 0)

    # get a list of names for all the devices plugged into the system
    devices = fx.enumerate_devices()
    if (len(devices) < 1):
        print("Error - no NCS devices detected, verify an NCS device is connected.")
        quit() 


    # get the first NCS device by its name.  For this program we will always open the first NCS device.
    dev = fx.Device(devices[0])

    
    # try to open the device.  this will throw an exception if someone else has it open already
    try:
        dev.open()
    except:
        print("Error - Could not open NCS device.")
        quit()


    print("Hello NCS! Device opened normally.")
    

    with open('output/laneseg.graph', mode='rb') as f:
        graphFileBuff = f.read()

    graph = fx.Graph('graph')

    print("FIFO Allocation")
    fifoIn, fifoOut = graph.allocate_with_fifos(dev, graphFileBuff)

    import numpy as np

    img = np.zeros((160, 320, 3), dtype='float32')

    import time
    start = time.time()

    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img, 'user object')
    output, userobj = fifoOut.read_elem()

    print(time.time() - start)
    print(output.shape, userobj)

    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()

    try:
        dev.close()
    except:
        print("Error - could not close NCS device.")
        quit()

    print("Goodbye NCS! Device closed normally.")
    print("NCS device working.")