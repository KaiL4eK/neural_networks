#include "yolo_trt.hpp"

#include <iostream>
using namespace std;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


#include "NvUffParser.h"
#include "NvInferPlugin.h"

#include "buffers.h"
// #include "logger.h"
#include "common.h"

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

Logger gLogger(Logger::Severity::kINFO);

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

// class Logger : public nv::ILogger
// {
// public:
//     void log(nv::ILogger::Severity severity, const char* msg) override
//     {
//         switch (severity)
//         {
//             case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
//             case Severity::kERROR: std::cerr << "ERROR: "; break;
//             case Severity::kWARNING: std::cerr << "WARNING: "; break;
//             case Severity::kINFO: std::cerr << "INFO: "; break;
//             default: std::cerr << "UNKNOWN: "; break;
//         }
//         std::cerr << msg << std::endl;
//     }
// };

YOLO_TensorRT::YOLO_TensorRT(std::string cfg_fpath) : 
    CommonYOLO(cfg_fpath)
{
}

bool YOLO_TensorRT::init(std::string uff_fpath, bool fp16_enabled)
{
    // Logger logger;
    // initLibNvInferPlugins(&logger, "");

    fs::path enginePath = fs::path(uff_fpath).filename().replace_extension(".tngn");

    if ( fs::exists(enginePath) )
    {
        std::ifstream ifile(enginePath.string(), std::ios::binary | ios::ate);
        size_t serializedModelSz = ifile.tellg();
        char * serializedModelData = new char[serializedModelSz];

        ifile.seekg(0);

        ifile.read(serializedModelData, serializedModelSz);

        nv::IRuntime* runtime = nv::createInferRuntime(gLogger);
        mEngine = std::shared_ptr<nv::ICudaEngine>(runtime->deserializeCudaEngine(serializedModelData, serializedModelSz, nullptr), InferDeleter());
    }
    else
    {
        auto builder = UniquePtr<nv::IBuilder>(nv::createInferBuilder(gLogger));
        if (!builder)
        {
            return false;
        }

        auto network = UniquePtr<nv::INetworkDefinition>(builder->createNetwork());
        if (!network)
        {
            return false;
        }

        // For version 6.0
        // auto config = UniquePtr<nv::IBuilderConfig>(builder->createBuilderConfig());
        // if (!config)
        // {
            // return false;
        // }

        auto parser = UniquePtr<uff::IUffParser>(uff::createUffParser());
        if (!parser)
        {
            return false;
        }

        parser->registerInput(mCfg._input_names[0].c_str(),
                            nv::DimsCHW(3, mCfg._infer_sz.height, mCfg._infer_sz.width), 
                            uff::UffInputOrder::kNCHW);

        for ( const string &name : mCfg._output_names )
        {
            parser->registerOutput(name.c_str());
        }
        
        auto parsed = parser->parse(uff_fpath.c_str(), *network, nv::DataType::kFLOAT);
        if (!parsed)
        {
            return false;
        }
        
        assert(network->getNbInputs() == 1);
        nv::Dims inputDims = network->getInput(0)->getDimensions();
        assert(inputDims.nbDims == 3);

        builder->setMaxBatchSize(mCfg._tile_cnt);
        builder->setMaxWorkspaceSize(256_MB);
        // if (fp16_enabled)
        // {
            // builder->setFlag(nv::BuilderFlag::kFP16);
        // }

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

        mEngine = std::shared_ptr<nv::ICudaEngine>(builder->buildCudaEngine(*network), InferDeleter());
        if (!mEngine)
        {
            return false;
        }

        auto serializedModel = UniquePtr<nv::IHostMemory>(mEngine->serialize());
        void *serializedModelData = serializedModel->data();
        size_t serializedModelSz = serializedModel->size();

        std::ofstream ofile(enginePath.string(), std::ios::binary);
        ofile.write((char *)serializedModelData, serializedModelSz);
    }

    mBatchSize = mCfg._tile_cnt;
    mInputName = mCfg._input_names[0];

    for ( const string &name : mCfg._output_names )
    {
        mOutputNames.push_back(name);
    }

    return true;
}

void mat_2_blob(const cv::Mat &orig_image, float *blob, int batchIndex = 0)
{
    cv::Size input_sz = orig_image.size();

    const size_t width = input_sz.width;
    const size_t height = input_sz.height;
    const size_t channels = 3;
    float *blob_data = blob;

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < height; h++)
        {
            for (size_t w = 0; w < width; w++)
            {
                /* TODO - Isn`t is transposed? */
                blob_data[batchOffset + c * width * height + h * width + w] =
                    orig_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void YOLO_TensorRT::infer(cv::Mat raw_image, std::vector<DetectionObject> &detections)
{
    chrono::time_point<chrono::steady_clock> inf_start_time = chrono::steady_clock::now();
    
    samplesCommon::BufferManager buffers(mEngine, mCfg._tile_cnt);

    void *inputData_CHW = buffers.getHostBuffer(mInputName);

    // cout << mEngine->getBindingFormatDesc(mEngine->getBindingIndex(mInputName.c_str())) << endl;

    ImageResizeConfig   rsz_cfg;
    initResizeConfig(raw_image, rsz_cfg);

    cv::Mat inputFrame;
    resizeForNetwork(raw_image, inputFrame, rsz_cfg);

    inputFrame.convertTo(inputFrame, CV_32F);

    inputFrame = inputFrame/255.;

    cout << type2str( inputFrame.type() ) << endl;

    mat_2_blob(inputFrame, (float *)inputData_CHW);

    cout << "BufferManager created" << endl;

    auto context = UniquePtr<nv::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        cout << "Failed to create ExecutionContext" << endl;
        return;
    }

    cout << "ExecutionContext created" << endl;

    cout << "Size (bytes) of input buffer: " << buffers.size(mCfg._input_names[0]) << endl;

    buffers.copyInputToDevice();

    bool status = context->execute(1, buffers.getDeviceBindings().data());
    if (!status)
    {
        cout << "Status: failed execution" << endl;
        return;
    }
    
    cout << "Status: execution processed" << endl;

    buffers.copyOutputToHost();

    cout << "Processing output" << endl;

    vector<RawDetectionObject> global_dets;

    for ( size_t i = 0; i < mCfg._tile_cnt; i++ )
    {
        for (size_t i_layer = 0; i_layer < mOutputNames.size(); i_layer++)
        {
            void *output_blob = buffers.getHostBuffer(mOutputNames[i_layer]);
            
            int index = mEngine->getBindingIndex(mOutputNames[i_layer].c_str());
            nv::Dims outputDims = mEngine->getBindingDimensions(index);

            // cout << outputDims.nbDims << ": " << outputDims.d[0] << " / " << outputDims.d[1] << " / " << outputDims.d[2] << endl;
            // cout << mEngine->getBindingFormatDesc(index) << endl;
            
            const float grid_w = outputDims.d[1];
            const float grid_h = outputDims.d[0];
            const size_t chnl_count = outputDims.d[2];

            vector<cv::Point> anchors = get_anchors(i_layer);
            get_detections(global_dets, output_blob, grid_h, grid_w, chnl_count, anchors, ParsingFormat::HWC);
        }

        rsz_cfg.tile_idx = i;
        postprocessBoxes(global_dets, rsz_cfg);
    }

    filterBoxes(global_dets, detections);

    chrono::duration<double> inf_elapsed = chrono::steady_clock::now() - inf_start_time;
    cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(inf_elapsed).count() << " [ms]" << endl;

}
