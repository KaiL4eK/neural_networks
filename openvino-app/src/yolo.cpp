#include "yolo.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <chrono>
using namespace std;

/*
Python
    [0.575838:0.598583, 0.332272:0.353946] / classes: [0.999]]
    [0.572906:0.601659, 0.328150:0.356894] / classes: [0.916]]
    [0.581757:0.606235, 0.890164:0.912859] / classes: [0.999]]
    [0.578065:0.605363, 0.885544:0.915002] / classes: [0.999]]

WH - same, XY - different???

OpenVINO
    [0.546249:0.568993, 0.324181:0.345855] 0.871
    [0.543336:0.572089, 0.319987:0.348731] 0.679
    [0.557611:0.582089, 0.865725:0.888420] 0.920
    [0.550974:0.578271, 0.859379:0.888837] 0.888
*/

CommonYOLO::CommonYOLO(std::string cfg_path) :
    mCfg(cfg_path)
{
}


YOLOConfig::YOLOConfig(string cfg_path)
{
    pt::ptree cfg_root;
    pt::read_json(cfg_path, cfg_root);

    pt::ptree model_root = cfg_root.get_child("model");

    /* Read anchors */
    cv::Point anchors_pair(-1, -1);

    _output_cnt = model_root.get_child("downsample").size();

    for (pt::ptree::value_type &v : model_root.get_child("anchors"))
    {
        if (anchors_pair.x < 0)
        {
            anchors_pair.x = v.second.get_value<uint32_t>();
        }
        else
        {
            anchors_pair.y = v.second.get_value<uint32_t>();
            _anchors.push_back(anchors_pair);
            anchors_pair.x = -1; /* reset to read next number */
        }
    }

    vector<size_t> infer_size_raw;
    for (pt::ptree::value_type &v : model_root.get_child("infer_shape"))
    {
        infer_size_raw.push_back(v.second.get_value<uint32_t>());
    }

    if ( infer_size_raw.size() != 2 )
        throw invalid_argument("Invalid field 'model/infer_shape'");

    _infer_sz = cv::Size(infer_size_raw[1], infer_size_raw[0]);

    cout << "** Config **" << endl;
    cout << "Readed anchors: " << endl;
    for (cv::Point &pnt : _anchors)
    {
        cout << "  " << pnt << endl;
    }

    /* Read tile count */
    _tile_cnt = model_root.get_child("tiles").get_value<uint32_t>();

    cout << "Readed tiles count: " << _tile_cnt << endl;

    /* TODO - disable hardlink */
    _objectness_thresh = 0.5;
}

std::vector<cv::Point> CommonYOLO::get_anchors(size_t layer_idx)
{
    vector<cv::Point> anchors;

    size_t anchors_per_output = mCfg._anchors.size() / mCfg._output_cnt;
    size_t start_idx = anchors_per_output * (mCfg._output_cnt - layer_idx - 1);
    size_t end_idx = anchors_per_output * (mCfg._output_cnt - layer_idx);
    
    cout << start_idx << " / " << end_idx << endl;

    for ( size_t i = start_idx; i < end_idx; i++ )
    {
        anchors.push_back(mCfg._anchors[i]);
    }

    return anchors;
}

void CommonYOLO::initResizeConfig(cv::Mat in_img, 
                                  ImageResizeConfig &cfg)
{
    uint32_t new_w, new_h;

    cv::Size2f tile_sz;

    if (mCfg._tile_cnt == 1)
    {
        tile_sz = cv::Size2f(in_img.cols, in_img.rows);
        cfg.tile_rects.push_back( cv::Rect(cv::Point(0, 0), tile_sz) );
    }
    else if (mCfg._tile_cnt == 2)
    {
        tile_sz = cv::Size(in_img.cols/2, in_img.rows);
        cfg.tile_rects.push_back( cv::Rect(cv::Point(0, 0), tile_sz) );
        cfg.tile_rects.push_back( cv::Rect(cv::Point(in_img.cols/2, 0), tile_sz) );
    }

    if ( (mCfg._infer_sz.width / tile_sz.width) < (mCfg._infer_sz.height / tile_sz.height) )
    {
        new_w = mCfg._infer_sz.width;
        new_h = tile_sz.height / tile_sz.width * mCfg._infer_sz.width;
    }
    else
    {
        new_h = mCfg._infer_sz.height;
        new_w = tile_sz.width / tile_sz.height * mCfg._infer_sz.height;
    }

    cfg.top = (mCfg._infer_sz.height - new_h) / 2;
    cfg.bottom = (mCfg._infer_sz.height - new_h) - cfg.top;
    cfg.left = (mCfg._infer_sz.width - new_w) / 2;
    cfg.right = (mCfg._infer_sz.width - new_w) - cfg.left;

    cfg.new_sz = cv::Size(new_w, new_h);
    cfg.old_sz = in_img.size();

    cfg.offset = cv::Point2d(
        static_cast<float>(mCfg._infer_sz.width - cfg.new_sz.width) / 2. / mCfg._infer_sz.width,
        static_cast<float>(mCfg._infer_sz.height - cfg.new_sz.height) / 2. / mCfg._infer_sz.height
    );

    cfg.scale = cv::Point2d(
        static_cast<float>(cfg.new_sz.width) / mCfg._infer_sz.width,
        static_cast<float>(cfg.new_sz.height) / mCfg._infer_sz.height
    );
}

void CommonYOLO::resizeForNetwork(cv::Mat in_img, 
                                  cv::Mat &out_img,
                                  ImageResizeConfig &cfg)
{
    cv::Mat tile_img = get_roi_tile(in_img, cfg.tile_idx);

    cv::resize(tile_img, tile_img, cfg.new_sz);

    cv::copyMakeBorder(tile_img, out_img, 
                    cfg.top, cfg.bottom, cfg.left, cfg.right, 
                    cv::BORDER_CONSTANT, 
                    cv::Scalar(127, 127, 127));
}

void CommonYOLO::postprocessBoxes(std::vector<RawDetectionBox> &raw_boxes,
                                  std::vector<DetectionBox> &result_boxes,
                                  ImageResizeConfig &cfg)
{
    /* For correction */


    for ( RawDetectionBox &det : raw_boxes )
    {
        // cv::Point tl(
        //     (det.box_x - det.box_w/2) * mInferSize.width,
        //     (det.box_y - det.box_h/2) * mInferSize.height);
        // cv::Point br(
        //     (det.box_x + det.box_w/2) * mInferSize.width,
        //     (det.box_y + det.box_h/2) * mInferSize.height);

        // cout << tl << " / " << br << endl;

        // cv::rectangle(net_input_frame, tl, br, cv::Scalar(250, 0, 0), 2);

        cout << "[" 
                << det.box_y-det.box_h/2 << ":" 
                << det.box_y+det.box_h/2 << ", " 
                << det.box_x-det.box_w/2 << ":" 
                << det.box_x+det.box_w/2 
                << "] " 
                << det.cls << endl; 

        DetectionBox px_det;
        px_det.cls = det.cls;
        px_det.cls_idx = det.cls_idx;

        float box_x = (det.box_x - cfg.offset.x) / cfg.scale.x * cfg.tile_rects[cfg.tile_idx].width;
        float box_y = (det.box_y - cfg.offset.y) / cfg.scale.y * cfg.tile_rects[cfg.tile_idx].height;
        float box_w = det.box_w / cfg.scale.x * cfg.tile_rects[cfg.tile_idx].width;
        float box_h = det.box_h / cfg.scale.y * cfg.tile_rects[cfg.tile_idx].height;

        box_x += cfg.tile_rects[cfg.tile_idx].x;
        box_y += cfg.tile_rects[cfg.tile_idx].y;
        
        px_det.rect = cv::Rect(
            cv::Point( box_x - box_w/2, box_y - box_h/2 ),
            cv::Point( box_x + box_w/2, box_y + box_h/2 )
        );

        result_boxes.push_back(px_det);
    }
}

cv::Mat CommonYOLO::get_roi_tile(cv::Mat raw_image, size_t idx)
{
    if (mCfg._tile_cnt == 1)
    {
        return raw_image;
    }
    else if (mCfg._tile_cnt == 2)
    {
        if ( idx == 0 )
        {
            cv::Rect roi_left(cv::Point(0, 0), cv::Point(raw_image.cols/2, raw_image.rows));
            return raw_image(roi_left);
        }
        else if ( idx == 1 )
        {
            cv::Rect roi_right(cv::Point(raw_image.cols/2, 0), cv::Point(raw_image.cols, raw_image.rows));
            return raw_image(roi_right);
        }
    }
}

