

TRT_ROOT = $(HOME)/data/nvidia/TensorRT-4.0.1.6
INCLUDES = -I $(TRT_ROOT)/include -I src -I $(PYTHON_HEADER_PATH)

all:
	swig -c++ -python src/tensorNet.i 
	nvcc -std=c++11 -O3 --compiler-options '-fPIC' -c src/tensorNet.cpp $(INCLUDES)
	nvcc -std=c++11 -O3 --compiler-options '-fPIC' -c src/tensorNet_wrap.cxx $(INCLUDES)
	nvcc -shared tensorNet.o tensorNet_wrap.o -lnvinfer -lnvparsers -lnvinfer_plugin -o src/_tensorNet.so -L $(TRT_ROOT)/lib
	rm *.o
