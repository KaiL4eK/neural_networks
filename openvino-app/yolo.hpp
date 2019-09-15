#pragma once

#include <vector>

#include <opencv2/core.hpp>


struct YOLOConfig
{
    YOLOConfig(std::string cfg_path);

    std::vector<cv::Point>  _anchors;
    uint32_t                _tile_cnt;
};

struct DetectionBox
{
    cv::Rect px_rect;
    uint32_t cls_idx;
};

class YOLONetwork
{
public:
    YOLONetwork(YOLOConfig cfg, cv::Size infer_sz);

    void infer(cv::Mat frame, std::vector<DetectionBox> &detections);

private:
    void preprocess(cv::Mat in_frame, cv::Mat out_frame);

    YOLOConfig  mCfg;
    cv::Size    mInferSize;
};
