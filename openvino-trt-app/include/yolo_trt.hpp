#pragma once

#include "yolo.hpp"

class YOLO_TensorRT : public CommonYOLO
{
public:
    YOLO_TensorRT(std::string cfg_fpath);    

    void init(std::string uff_fpath, bool fp16_enabled);

    void infer(cv::Mat raw_image, std::vector<DetectionObject> &detections) override;

private:
    std::string                                         mInputName;
    std::vector<std::string>                            mOutputNames;
};
