#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <cmath>
#include <iostream>
using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>

#include "yolo_trt.hpp"

string g_uff_path;
string g_cfg_path;
bool g_fp16_enabled = false;
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
    po::options_description desc("Options for detection app");

    po::options_description_easy_init opt_init = desc.add_options();
    opt_init("help,h", "Print help messages");
    opt_init("uff_path,r", po::value<string>(), "UFF path");
    opt_init("cfg_path,c", po::value<string>(), "Net config path");
    opt_init("fp16", "Enable FP16 mode");
    opt_init("image,i", po::value<string>(), "Path to sample image");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm); // can throw

        if (vm.count("help"))
        {
            cout << "Basic Command Line Parameter App" << endl
                 << desc << endl;
            return EXIT_SUCCESS;
        }

        if (vm.count("fp16"))
        {
            g_fp16_enabled = true;
            cout << "FP16 mode enabled" << endl;
        }

        if (vm.count("image"))
        {
            g_image_fpath = vm["image"].as<string>();

            cout << "Set image path: " << g_image_fpath << endl;
        }

        if (vm.count("uff_path"))
        {
            g_uff_path = vm["uff_path"].as<string>();

            cout << "Parsing UFF path: " << g_uff_path << endl;
        }
        else
        {
            cout << "UFF path not set, reset" << endl;
            cout << desc << endl;
            return EXIT_FAILURE;
        }

        if (vm.count("cfg_path"))
        {
            g_cfg_path = vm["cfg_path"].as<string>();

            cout << "Parsing Config path: " << g_cfg_path << endl;
        }
        else
        {
            cout << "Config path not set, reset" << endl;
            cout << desc << endl;
            return EXIT_FAILURE;
        }

        po::notify(vm);
    }
    catch (po::error &e)
    {
        cerr << "Error: " << e.what() << endl
             << endl;
        cerr << desc << endl;
        return EXIT_FAILURE;
    }

    cv::Mat input_image = cv::imread(g_image_fpath);

    YOLO_TensorRT yolo(g_cfg_path);
    yolo.init(g_uff_path, g_fp16_enabled);

    std::vector<DetectionObject> corrected_dets;
    yolo.infer(input_image, corrected_dets);

    for (DetectionObject &det : corrected_dets)
    {
        cv::rectangle(input_image, det.rect, cv::Scalar(250, 0, 0), 2);
    }

    cv::imshow("Boxes", input_image);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
