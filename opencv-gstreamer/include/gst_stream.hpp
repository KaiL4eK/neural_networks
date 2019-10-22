#pragma once

#include <opencv2/videoio.hpp>

class TCPStreamGST
{
public:
    TCPStreamGST();

    void open_cap(int port);
    void send_frame(cv::Mat frame);

private:


    cv::VideoWriter     mWriter;
};

// class RTSPStreamGST
// {
// public:

//     RTSPStreamGST();

// private:
//     void start_cast();
// }
