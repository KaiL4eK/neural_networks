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

    cv::Mat get_input(cv::Mat raw_image, size_t idx);
    void correct_detections(cv::Mat raw_image, std::vector<std::vector<RawDetectionBox>> &raw_dets, std::vector<DetectionBox> &corrected_dets);

    std::vector<cv::Point> get_anchors(size_t layer_idx);

    size_t get_infer_count() { return mCfg._tile_cnt; }

private:
    cv::Mat preprocess(cv::Mat in_frame);

    YOLOConfig  mCfg;
    cv::Size2f  mInferSize;
};
