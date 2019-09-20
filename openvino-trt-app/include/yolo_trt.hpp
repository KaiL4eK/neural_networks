#pragma once

#include "yolo.hpp"

#include "NvInfer.h"



class YOLO_TensorRT : public CommonYOLO
{
public:
    YOLO_TensorRT(std::string cfg_fpath);    

    bool init(std::string uff_fpath, bool fp16_enabled);

    void infer(cv::Mat raw_image, std::vector<DetectionObject> &detections) override;

private:

    std::shared_ptr<nvinfer1::ICudaEngine>              mEngine;

    std::string                                         mInputName;
    std::vector<std::string>                            mOutputNames;
};
