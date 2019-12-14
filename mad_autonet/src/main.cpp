#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <cmath>
#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "yolo_ov.hpp"

#include <ros/ros.h>

string g_ir_path;
string g_cfg_path;
string g_device_type;
string g_image_fpath;

// ostream &operator<<(ostream &out, const vector<size_t> &c)
// {
//     out << "[";
//     for (const size_t &s : c)
//         out << s << ",";
//     out << "]";
//     return out;
// }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle nh("~");

    nh.getParam("config_path", g_cfg_path);
    nh.getParam("ir_path", g_ir_path);
    nh.getParam("device", g_device_type);
    nh.getParam("input", g_image_fpath);

    cv::Mat input_image = cv::imread(g_image_fpath);

    YOLO_OpenVINO yolo(g_cfg_path);
    yolo.init(g_ir_path, g_device_type);

    std::vector<DetectionObject> corrected_dets;
    yolo.infer(input_image, corrected_dets);

    for (DetectionObject &det : corrected_dets)
    {
        cv::rectangle(input_image, det.rect, cv::Scalar(250, 0, 0), 2);

        cout << "Detection: " << det.cls_idx << endl;
    }

    cv::Mat resized;
    cv::resize(input_image, resized, cv::Size(800, 600));

    cv::imshow("Boxes", resized);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
