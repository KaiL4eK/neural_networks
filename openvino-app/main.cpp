#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <cmath>
#include <iostream>
#include <array>
using namespace std;

#include <ext_list.hpp>
#include <inference_engine.hpp>
namespace ie = InferenceEngine;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>

/* Taken from intel samples */
#include "slog.hpp"
#include "yolo.hpp"

string g_ir_path;
string g_cfg_path;
string g_device_type;
string g_image_fpath;

ostream &operator<<(ostream &out, const vector<size_t> &c)
{
    out << "[";
    for (const size_t &s : c)
        out << s << ",";
    out << "]";
    return out;
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

int main(int argc, char **argv)
{
    po::options_description desc("Options for detection app");

    po::options_description_easy_init opt_init = desc.add_options();
    opt_init("help,h", "Print help messages");
    opt_init("ir_path,r", po::value<string>(), "IR path");
    opt_init("cfg_path,c", po::value<string>(), "Net config path");
    opt_init("device,d", po::value<string>()->default_value("CPU"), "Device type: CPU, MYRIAD");
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

        if (vm.count("device"))
        {
            g_device_type = vm["device"].as<string>();

            cout << "Device type: " << g_device_type << endl;
        }

        if (vm.count("image"))
        {
            g_image_fpath = vm["image"].as<string>();

            cout << "Set image path: " << g_image_fpath << endl;
        }

        if (vm.count("ir_path"))
        {
            g_ir_path = vm["ir_path"].as<string>();

            cout << "Parsing IR path: " << g_ir_path << endl;
        }
        else
        {
            cout << "IR path not set, reset" << endl;
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

    cout << "InferenceEngine: " << ie::GetInferenceEngineVersion() << endl;
    cout << "Loading Inference Engine" << endl;
    ie::Core ie_core;

    // cout << ie_core.GetVersions(g_device_type) << endl;

    if (g_device_type == "CPU")
    {
        ie_core.AddExtension(make_shared<ie::Extensions::Cpu::CpuExtensions>(), "CPU");
    }

    string ir_bin_path = fs::path(g_ir_path).replace_extension(".bin").string();

    cout << "Loading network files:\n\t"
         << g_ir_path << "\n\t"
         << ir_bin_path << endl;

    ie::CNNNetReader net_reader;
    net_reader.ReadNetwork(g_ir_path);
    net_reader.ReadWeights(ir_bin_path);
    ie::CNNNetwork network = net_reader.getNetwork();

    cout << "Preparing input blobs" << endl;

    ie::InputsDataMap net_inputs_info(network.getInputsInfo());

    ie::InputInfo::Ptr &input_data = net_inputs_info.begin()->second;
    input_data->setPrecision(ie::Precision::U8);

    const ie::SizeVector input_dims = input_data->getTensorDesc().getDims();
    cv::Size input_size(input_dims[3] /* w */, input_dims[2] /* h */);

    vector<string> outputNames;

    ie::OutputsDataMap net_outputs_info(network.getOutputsInfo());
    cout << "Output names: " << endl;
    for (const auto &out : net_outputs_info)
    {
        const ie::SizeVector dims = out.second->getTensorDesc().getDims();

        // out.second->setPrecision(ie::Precision::FP16);

        cout << "   " << out.first
             << " / " << dims
             << " / " << out.second->getPrecision()
             << endl;

        outputNames.push_back(out.first);
    }

    YOLOConfig main_config(g_cfg_path);

    cout << "Loading to device" << endl;
    ie::ExecutableNetwork executable_network = ie_core.LoadNetwork(network, g_device_type);

    cv::Mat input_image = cv::imread(g_image_fpath);
    cout << input_image.size() << endl;

    YOLONetwork yolo(main_config, input_size);

    std::vector<DetectionBox> corrected_dets;
    yolo.infer(input_image, executable_network, corrected_dets);

    for (DetectionBox &det : corrected_dets)
    {
        cv::Point tl(
            (det.box_x - det.box_w/2),
            (det.box_y - det.box_h/2));
        cv::Point br(
            (det.box_x + det.box_w/2),
            (det.box_y + det.box_h/2));

        cout << tl << " / " << br << endl;

        cv::rectangle(input_image, tl, br, cv::Scalar(250, 0, 0), 2);
    }

    cv::imshow("Boxes", input_image);
    cv::waitKey(0);


    return EXIT_SUCCESS;
}
