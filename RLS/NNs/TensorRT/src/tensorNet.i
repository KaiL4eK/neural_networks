%module tensorNet

%{
    #define SWIG_FILE_WITH_INIT
    #include "tensorNet.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3) 
		{(int chnls, int rows, int cols, float* data_in)}
%apply (int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3) 
		{(int rows, int cols, int chnls, float* data_out)}

%include "tensorNet.h"
