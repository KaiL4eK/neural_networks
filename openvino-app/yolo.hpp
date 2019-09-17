#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <inference_engine.hpp>

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
    float cls;
    size_t cls_idx;
};

struct DetectionBox
{
    int box_x;
    int box_y;
    int box_w;
    int box_h;
    float cls;
    size_t cls_idx;
};



class YOLONetwork
{
public:
    YOLONetwork(YOLOConfig cfg, cv::Size infer_sz);

    std::vector<cv::Point> get_anchors(size_t layer_idx);
    
    size_t get_infer_count() { return mCfg._tile_cnt; }

    void infer(cv::Mat raw_image, 
               InferenceEngine::ExecutableNetwork &executable_network, 
               std::vector<DetectionBox> &detections);

private:
    cv::Mat get_roi_tile(cv::Mat raw_image, size_t idx);

    YOLOConfig  mCfg;
    cv::Size2f  mInferSize;
};
