#include "yolo_trt.hpp"

using namespace std;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"

YOLO_TensorRT::YOLO_TensorRT(std::string cfg_fpath) : 
    CommonYOLO(cfg_fpath)
{
}

void YOLO_TensorRT::init(std::string uff_fpath, bool fp16_enabled)
{
    initLibNvInferPlugins(NULL, "");
}

void YOLO_TensorRT::infer(cv::Mat raw_image, std::vector<DetectionObject> &detections)
{

}
