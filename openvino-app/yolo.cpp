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

    cout << "** Config **" << endl;
    cout << "Readed anchors: " << endl;
    for (cv::Point &pnt : _anchors)
    {
        cout << "  " << pnt << endl;
    }

    /* Read tile count */
    _tile_cnt = model_root.get_child("tiles").get_value<uint32_t>();

    cout << "Readed tiles count: " << _tile_cnt << endl;
}

YOLONetwork::YOLONetwork(YOLOConfig cfg, cv::Size infer_sz) : mCfg(cfg), mInferSize(infer_sz)
{
}

cv::Mat YOLONetwork::preprocess(cv::Mat in_frame)
{
    // cv::Size inferSize = mInferSize;

    // if ( mCfg._tile_cnt == 2 )
    // {
    //     inferSize.width *= 2;
    // }

    // cout << inferSize << endl;
    // cout << in_frame.size() << endl;

    cv::Mat resizedFrame;

    cv::Mat paddedFrame;

    uint32_t top = 0,
             bottom = 0,
             left = 0,
             right = 0;

    double fx, fy;

    if ( abs(in_frame.cols-mInferSize.width) > abs(in_frame.rows-mInferSize.height) )
    {
        fx = fy = mInferSize.width * 1.0 / in_frame.cols;
    }
    else
    {
        fx = fy = mInferSize.height * 1.0 / in_frame.rows;
    }

    cv::resize(in_frame, resizedFrame, cv::Size(), fx, fy);

    if (in_frame.cols * 1.0 / in_frame.rows > 1)
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

    cv::copyMakeBorder(resizedFrame, paddedFrame, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

    // cout << paddedFrame.size() << endl;
    // cv::imshow("Processed", paddedFrame);
    // cv::imshow("Original", in_frame);
    // cv::waitKey(0);

    return paddedFrame;
}

std::vector<cv::Point> YOLONetwork::get_anchors(size_t layer_idx)
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

void YOLONetwork::get_inputs(cv::Mat raw_image, std::vector<cv::Mat> &inputs)
{
    inputs.reserve( mCfg._tile_cnt );

    if (mCfg._tile_cnt == 1)
    {
        inputs.insert(inputs.begin(), preprocess(raw_image));
    }
    else if (mCfg._tile_cnt == 2)
    {
        cv::Rect roi_left(cv::Point(0, 0), cv::Point(raw_image.cols/2, raw_image.rows));
        cv::Rect roi_right(cv::Point(raw_image.cols/2, 0), cv::Point(raw_image.cols, raw_image.rows));

        inputs.insert(inputs.begin(), preprocess(raw_image(roi_left)));
        inputs.insert(inputs.begin()+1, preprocess(raw_image(roi_right)));
    }
    else
    {
        throw logic_error("Invalid tiles count: " + to_string(mCfg._tile_cnt));
    }
}
