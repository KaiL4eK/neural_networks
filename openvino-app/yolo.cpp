#include "yolo.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace std;

YOLOConfig::YOLOConfig(string cfg_path)
{
    pt::ptree cfg_root;
    pt::read_json(cfg_path, cfg_root);

    pt::ptree model_root = cfg_root.get_child("model");

    /* Read anchors */
    cv::Point anchors_pair(-1, -1);
    for ( pt::ptree::value_type &v : model_root.get_child("anchors") )
    {
        if ( anchors_pair.x < 0 )
        {
            anchors_pair.x = v.second.get_value<uint32_t>();
        }
        else
        {
            anchors_pair.y = v.second.get_value<uint32_t>();\
            _anchors.push_back(anchors_pair);
            anchors_pair.x = -1;    /* reset to read next number */
        }
    }

    cout << "** Config **" << endl;
    cout << "Readed anchors: " << endl;
    for ( cv::Point &pnt : _anchors )
    {
        cout << "  " << pnt << endl;
    }

    /* Read tile count */
    _tile_cnt = model_root.get_child("tiles").get_value<uint32_t>();

    cout << "Readed tiles count: " << _tile_cnt << endl;
}


YOLONetwork::YOLONetwork(YOLOConfig cfg, cv::Size infer_sz) :
    mCfg(cfg), mInferSize(infer_sz)
{
    
}

void YOLONetwork::preprocess(cv::Mat in_frame, cv::Mat out_frame)
{
    cv::Mat resizedFrame;

    cv::Mat paddedFrame;

    uint32_t top = 0,
             bottom = 0,
             left = 0,
             right = 0;

    double fx, fy;

    if ( in_frame.cols * 1.0 / in_frame.rows > 1 )
    {
        fx = fy = mInferSize.width * 1.0 / in_frame.cols;
    }
    else
    {
        fx = fy = mInferSize.height * 1.0 / in_frame.rows;        
    }
    
    cv::resize(in_frame, resizedFrame, cv::Size(), fx, fy);

    if ( in_frame.cols * 1.0 / in_frame.rows > 1 )
    {
        left = right = 0;
        top = (mInferSize.height - resizedFrame.rows) / 2;
        bottom = mInferSize.height - (resizedFrame.rows + top);
    }
    else
    {    
        top = bottom = 0;
        left = (mInferSize.width - resizedFrame.cols) / 2;
        right = mInferSize.width - (resizedFrame.cols + left);
    }

    cv::copyMakeBorder(resizedFrame, paddedFrame, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127) );

    out_frame = paddedFrame;
    cout << out_frame.size() << endl;

    cv::imshow("Processed", out_frame);
    
    cv::imshow("Original", in_frame);
    cv::waitKey(0);
}

void YOLONetwork::infer(cv::Mat frame, vector<DetectionBox> &detections)
{
    cv::Mat processed;
    preprocess(frame, processed);

    
}
