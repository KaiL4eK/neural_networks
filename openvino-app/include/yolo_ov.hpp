#pragma once

#include "yolo.hpp"
#include <inference_engine.hpp>

class YOLO_OpenVINO : public CommonYOLO
{
public:
    YOLO_OpenVINO(std::string cfg_fpath);    

    void init(InferenceEngine::ExecutableNetwork &executable_network);

    void infer(cv::Mat raw_image, 
               InferenceEngine::ExecutableNetwork &executable_network, 
               std::vector<DetectionBox> &detections);

private:
    std::string                                         mInputName;
    std::vector<std::string>                            mOutputNames;
    std::vector<InferenceEngine::InferRequest::Ptr>     mInferRequests;
};
