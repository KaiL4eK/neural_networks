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
    float                   _iou_threshold;
};

struct RawDetectionObject
{
    RawDetectionObject() :
        corrected(false)
    {

    }

    /* Center + size */
    float x;
    float y;
    float w;
    float h;

    float xm;
    float ym;
    float conf;
    size_t cls_idx;

    bool corrected;

    bool operator <(const RawDetectionObject &s2) const {
        return this->conf < s2.conf;
    }

    bool operator >(const RawDetectionObject &s2) const {
        return this->conf < s2.conf;
    }
};

struct DetectionObject
{
    /* TL + size */
    cv::Rect rect;
    float conf;
    size_t cls_idx;
};

double IntersectionOverUnion(const RawDetectionObject &box_1, 
                             const RawDetectionObject &box_2);

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

    virtual void infer(cv::Mat raw_image, std::vector<DetectionObject> &detections) = 0;

protected:
    void initResizeConfig(cv::Mat in_img, 
                          ImageResizeConfig &cfg);

    void resizeForNetwork(cv::Mat in_img, 
                          cv::Mat &out_img, 
                          ImageResizeConfig &cfg);

    /*
        Make postprocessing of boxes (net output -> image)
    */
    void postprocessBoxes(std::vector<RawDetectionObject> &raw_boxes,
                          ImageResizeConfig &cfg);

    void filterBoxes(std::vector<RawDetectionObject> &raw_boxes,
                     std::vector<DetectionObject> &result_boxes);

    cv::Mat get_roi_tile(cv::Mat raw_image, size_t idx);

    YOLOConfig  mCfg;
};

static inline double sigmoid(double x)
{
    // return x / (1.0 + fabs(x));
    return 1/(1+exp(-x));
}

// void softmax(float *classes, size_t sz)
// {
//     float sum = 0;

//     for (size_t i = 0; i < sz; i++)
//     {
//         classes[i] = exp(classes[i]);
//         sum += classes[i];
//     }

//     for (size_t i = 0; i < sz; i++)
//     {
//         classes[i] = classes[i] / sum;
//     }
// }
