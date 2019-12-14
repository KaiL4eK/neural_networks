#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/algorithm/string/predicate.hpp>

#include <cmath>
#include <iostream>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "yolo_ov.hpp"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

string g_ir_path;
string g_cfg_path;
string g_device_type;
string g_input;

// ostream &operator<<(ostream &out, const vector<size_t> &c)
// {
//     out << "[";
//     for (const size_t &s : c)
//         out << s << ",";
//     out << "]";
//     return out;
// }

class FramesSource
{
public:
    virtual cv::Mat get_frame();

protected:
    cv::Mat frame_;

    mutex               frame_mutex_;
    condition_variable  frame_condvar_;
    bool is_empty_ = true;
};

cv::Mat FramesSource::get_frame()
{
    unique_lock<mutex> lock(frame_mutex_);

    while ( is_empty_ ) {
        frame_condvar_.wait(lock);
    }

    is_empty_ = true;
    return frame_;
}

class VideoFramesSource : public FramesSource
{

};

class PictureFramesSource : public FramesSource
{

};

class RosTopicFramesSource : public FramesSource
{
public:
    RosTopicFramesSource(ros::NodeHandle &nh, string &topic_name);

private:
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;

    thread  polling_thread_;         

    void polling_routine();
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
};

RosTopicFramesSource::RosTopicFramesSource(ros::NodeHandle &nh, string &topic_name) :
    it_(nh)
{
    ROS_INFO_STREAM("Subscribing to " << topic_name);

    sub_ = it_.subscribe(topic_name, 1, &RosTopicFramesSource::image_callback, this);

    polling_thread_ = thread(&RosTopicFramesSource::polling_routine, this);
}

void RosTopicFramesSource::polling_routine()
{
    ROS_INFO_STREAM("Polling thread started");
    ros::spin();
}

void RosTopicFramesSource::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    unique_lock<mutex> lock(frame_mutex_);
    
    try
    {
        frame_ = cv_bridge::toCvShare(msg, "bgr8")->image;

        is_empty_ = false;
        frame_condvar_.notify_all();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle pr_nh("~");
    ros::NodeHandle nh;

    pr_nh.getParam("config_path", g_cfg_path);
    pr_nh.getParam("ir_path", g_ir_path);
    pr_nh.getParam("device", g_device_type);
    pr_nh.getParam("input", g_input);

    shared_ptr<FramesSource> source;

    if ( boost::starts_with(g_input, "/dev") ) {

    } else if ( boost::ends_with(g_input, ".mp4") ) {

    } else if ( boost::ends_with(g_input, ".png") ) {

    } else if ( boost::ends_with(g_input, ".jpg") ) {
        
    } else {
        source = make_shared<RosTopicFramesSource>(nh, g_input);
    }

    YOLO_OpenVINO yolo(g_cfg_path);
    yolo.init(g_ir_path, g_device_type);

    while ( ros::ok() )
    {
        cv::Mat input_image = source->get_frame();

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
        cv::waitKey(20);
    }

    return EXIT_SUCCESS;
}
