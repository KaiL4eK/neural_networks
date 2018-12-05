#include "tensorNet.h"
#include <NvUffParser.h>
#include <cuda_runtime.h>

using namespace nvuffparser;
using namespace nvinfer1;

#define MAX_WORKSPACE (1 << 30)

size_t getBufferSize(Dims d, DataType t)
{
    size_t size = 1;
    
    for(size_t i=0; i<d.nbDims; i++) 
        size*= d.d[i];

    switch (t) {
        case DataType::kFLOAT: return size*4;
        case DataType::kHALF: return size*2;
        case DataType::kINT8: return size*1;
    }

    assert(0);
    return 0;
}

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <fstream>

using namespace std;

static void create_buffer( MyEngine *my_engine )
{
    ICudaEngine *engine = my_engine->m_engine;

    my_engine->context = engine->createExecutionContext();

    std::cout << "[cpp] Create CUDA buffers" << std::endl;

    int nbBindings = engine->getNbBindings();

    my_engine->buffers.clear();
    my_engine->buffers.reserve(nbBindings);

    for( int i=0; i<nbBindings; i++ ) 
    {
        size_t buffer_sz = getBufferSize(engine->getBindingDimensions(i),  engine->getBindingDataType(i));
        std::cout << "[cpp] Create CUDA buffer for " << to_string(buffer_sz) << " bytes" << std::endl;
        cudaMallocManaged(&my_engine->buffers[i],  buffer_sz);
    }

    std::cout << "[cpp] Successfully create binding buffer" << std::endl;
}

MyEngine* createTrtFromUFF(char* modelpath)
{
    MyEngine *my_engine = new MyEngine();

    auto parser = createUffParser();

    parser->registerInput("input_img", DimsCHW(3, 416, 416), UffInputOrder::kNHWC);

    parser->registerOutput("conv2d_10/BiasAdd");
    parser->registerOutput("conv2d_13/BiasAdd");

    IBuilder* builder = createInferBuilder(my_engine->gLogger);
    INetworkDefinition* network = builder->createNetwork();

    if (!parser->parse(modelpath, *network, nvinfer1::DataType::kFLOAT)) 
    {
        std::cout << "[ChatBot] Fail to parse UFF model " << modelpath << std::endl;
        exit(0);
    }

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine) {
        std::cout << "[ChatBot] Unable to create engine" << std::endl;
        exit(0);
    }

    network->destroy();
    builder->destroy();
    parser->destroy();

    std::cout << "[cpp] Successfully create TensorRT engine from file " << modelpath << std::endl;

    my_engine->m_engine = engine;

    create_buffer( my_engine );

    return my_engine;
}

MyEngine* createTrtFromPlan(char* modelpath)
{
    MyEngine *my_engine = new MyEngine();

    ifstream planFile(modelpath);
    stringstream planBuffer;
    
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();
    
    IRuntime *runtime = createInferRuntime(my_engine->gLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);

    std::cout << "[cpp] Successfully create TensorRT engine from file " << modelpath << std::endl;

    my_engine->m_engine = engine;

    create_buffer( my_engine );

    return my_engine;
}

void showEngineSummary(MyEngine* my_engine) 
{
    ICudaEngine *engine = my_engine->m_engine;

    std::stringstream summary;

    for (int i = 0; i < engine->getNbBindings(); ++i) 
    {
        Dims dims = engine->getBindingDimensions(i);
        DataType dtype = engine->getBindingDataType(i);

        summary << "--Binding " << i << "--" << std::endl;
        if (engine->bindingIsInput(i))
            summary << "Type: Input";
        else
            summary << "Type: Output";
        summary << " DataType: ";
        if (dtype == DataType::kFLOAT)
            summary << "kFLOAT";
        else if (dtype == DataType::kHALF)
            summary << "kHALF";
        else if (dtype == DataType::kINT8)
            summary << "kINT8";

        summary << " Dims: (";
        for (int j = 0; j < dims.nbDims; j++)
            summary << dims.d[j] << ",";
        summary << ")" << std::endl;

    }

    std::cout << summary.str() << std::endl;
}

void inference(MyEngine* my_engine, int chnls, int rows, int cols, float* data_in)
{
    ICudaEngine *engine = my_engine->m_engine;

    if(!engine) 
    {
        std::cout << "[cpp] Invaild engine. Please remember to create engine first." << std::endl;
        exit(0);
    }

    Dims input_dims = engine->getBindingDimensions(0);

    assert( input_dims.nbDims == 3 );
    assert( input_dims.d[0] == chnls );
    assert( input_dims.d[1] == rows );
    assert( input_dims.d[2] == cols );

    size_t input_sz = getBufferSize(input_dims,  engine->getBindingDataType(0));

    cudaMemcpy(my_engine->buffers[0],
                data_in, input_sz,
                cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    assert(my_engine->context->execute(1, &my_engine->buffers[0]) == true);
    cudaThreadSynchronize();


}

void saveEngine(MyEngine* my_engine, char *filepath)
{
    ICudaEngine *engine = my_engine->m_engine;

    IHostMemory *engineSerialized = engine->serialize();

    std::ofstream engineSerializedFile;

    /*Open a new file for holding the serialized engine data*/
    engineSerializedFile.open(filepath, std::ios::out | std::ios::binary);

    if (engineSerializedFile.is_open() && engineSerializedFile.good() && !engineSerializedFile.fail())
    {
        /*Save the serialized engine data into the file*/
        engineSerializedFile.write(reinterpret_cast<const char *>(engineSerialized->data()), engineSerialized->size());

        /*Close the file*/
        engineSerializedFile.close();
    }
}

void getOutput(MyEngine* my_engine, int out_idx, int rows, int cols, int anchrs, int infos, float* data_out)
{
    ICudaEngine *engine = my_engine->m_engine;

    int bind_idx = out_idx + 1; /* 1 is input */

    Dims output_dims = engine->getBindingDimensions(bind_idx);

    assert( output_dims.nbDims == 3 );
    assert( output_dims.d[0] == rows );
    assert( output_dims.d[1] == cols );
    assert( output_dims.d[2] == anchrs * infos );

    size_t output_sz = getBufferSize(output_dims,  engine->getBindingDataType(0));

    cudaMemcpy(data_out,
                my_engine->buffers[bind_idx], output_sz,
                cudaMemcpyDeviceToHost);
}
