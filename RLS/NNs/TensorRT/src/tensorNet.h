#ifndef _TENSORNET_H_
#define _TENSORNET_H_

#include "NvInfer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

class MyEngine
{
public:
    MyEngine() : m_engine(nullptr) {}

    nvinfer1::ICudaEngine* m_engine;

    Logger gLogger;
    std::vector<void*> buffers;
    nvinfer1::IExecutionContext* context;

};

void saveEngine(MyEngine* my_engine, char *filepath);

MyEngine* createTrtFromUFF(char* uff_path, char* input, char* output);
MyEngine* createTrtFromPlan(char* modelpath);

void inference(MyEngine* engine, int chnls, int rows, int cols, float* data_in);

void getOutput(MyEngine* my_engine, int out_idx, int rows, int cols, int chnls, float* data_out);

void showEngineSummary(MyEngine* engine);

#endif // _TENSORNET_H_
