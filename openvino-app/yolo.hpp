#pragma once

#include <vector>

#include <opencv2/core.hpp>


struct YOLOConfig
{
    YOLOConfig(std::string cfg_path);

    std::vector<cv::Point>  _anchors;
    size_t                  _tile_cnt;
    size_t                  _output_cnt;
};

struct RawDetectionBox
{


    float box_x;
    float box_y;
    float box_w;
    float box_h;
    std::vector<float> cls;
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

    void get_inputs(cv::Mat raw_image, std::vector<cv::Mat> &inputs);

    std::vector<cv::Point> get_anchors(size_t layer_idx);

private:
    cv::Mat preprocess(cv::Mat in_frame);

    YOLOConfig  mCfg;
    cv::Size    mInferSize;
};
