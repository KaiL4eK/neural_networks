#include "yolo_trt.hpp"

#include <iostream>
using namespace std;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


#include "NvUffParser.h"
#include "NvInferPlugin.h"


// #include "logger.h"
// #include "common.h"

namespace nv = nvinfer1;
namespace uff = nvuffparser;


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger : public nv::ILogger
{
public:
    void log(nv::ILogger::Severity severity, const char* msg) override
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

YOLO_TensorRT::YOLO_TensorRT(std::string cfg_fpath) : 
    CommonYOLO(cfg_fpath)
{
}

bool YOLO_TensorRT::init(std::string uff_fpath, bool fp16_enabled)
{
    Logger logger;

    initLibNvInferPlugins(&logger, "");

    auto builder = UniquePtr<nv::IBuilder>(nv::createInferBuilder(logger));
    if (!builder)
    {
        return false;
    }

    auto network = UniquePtr<nv::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = UniquePtr<nv::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = UniquePtr<uff::IUffParser>(uff::createUffParser());
    if (!parser)
    {
        return false;
    }

    parser->registerInput(mCfg._input_names[0].c_str(),
                          nv::DimsCHW(3, mCfg._infer_sz.height, mCfg._infer_sz.width), 
                          uff::UffInputOrder::kNCHW);
    
    for ( const string &name : mCfg._output_names )
        parser->registerOutput(name.c_str());

    auto parsed = parser->parse(uff_fpath.c_str(), *network, nv::DataType::kFLOAT);
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mCfg._tile_cnt);
    config->setMaxWorkspaceSize(1 << 28 /* 256 MB */);
    if (fp16_enabled)
    {
        config->setFlag(nv::BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    // std::unique_ptr<IInt8Calibrator> calibrator;

    // if (mParams.int8)
    // {
    //     gLogInfo << "Using Entropy Calibrator 2" << std::endl;
    //     const std::string listFileName = "list.txt";
    //     const int imageC = 3;
    //     const int imageH = 300;
    //     const int imageW = 300;
    //     nv::DimsNCHW imageDims{};
    //     imageDims = nv::DimsNCHW{mParams.calBatchSize, imageC, imageH, imageW};
    //     BatchStream calibrationStream(
    //         mParams.calBatchSize, mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
    //     calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(
    //         calibrationStream, 0, "UffSSD", mParams.inputTensorNames[0].c_str()));
    //     config->setFlag(BuilderFlag::kINT8);
    //     config->setInt8Calibrator(calibrator.get());
    // }

    mEngine = std::shared_ptr<nv::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    nv::IHostMemory *serializedModel = mEngine->serialize();
    

    assert(network->getNbInputs() == 1);
    nv::Dims inputDims = network->getInput(0)->getDimensions();
    assert(inputDims.nbDims == 3);

    return true;
}

void YOLO_TensorRT::infer(cv::Mat raw_image, std::vector<DetectionObject> &detections)
{

}
