#pragma once

#include <vector>

#include <opencv2/core.hpp>


struct YOLOConfig
{
    YOLOConfig(std::string cfg_path);

    std::vector<cv::Point>  _anchors;
    size_t                  _tile_cnt;
    size_t                  _output_cnt;

    cv::Size                _infer_sz;

    float                   _objectness_thresh;
};

struct RawDetectionBox
{
    /* Center + size */
    float box_x;
    float box_y;
    float box_w;
    float box_h;
    float cls;
    size_t cls_idx;
};

struct DetectionBox
{
    /* TL + size */
    cv::Rect rect;
    float cls;
    size_t cls_idx;
};

struct ImageResizeConfig
{
    uint32_t top,
             bottom,
             left,
             right;
    
    cv::Size new_sz;
    cv::Size old_sz;

    std::vector<cv::Rect> tile_rects;

    cv::Point2d offset;
    cv::Point2d scale;

    size_t tile_idx;
};

class CommonYOLO
{
public:
    CommonYOLO(std::string cfg_fpath);

    size_t get_infer_count() { return mCfg._tile_cnt; }

    std::vector<cv::Point> get_anchors(size_t layer_idx);

protected:
    void initResizeConfig(cv::Mat in_img, 
                          ImageResizeConfig &cfg);

    void resizeForNetwork(cv::Mat in_img, 
                          cv::Mat &out_img, 
                          ImageResizeConfig &cfg);

    void postprocessBoxes(std::vector<RawDetectionBox> &raw_boxes,
                          std::vector<DetectionBox> &result_boxes,
                          ImageResizeConfig &cfg);

    cv::Mat get_roi_tile(cv::Mat raw_image, size_t idx);

    YOLOConfig  mCfg;
};

static inline double sigmoid(double x)
{
    // return x / (1.0 + fabs(x));
    return 1/(1+exp(-x));
}

