#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <iostream>
using namespace std;

#include <ext_list.hpp>
#include <inference_engine.hpp>
namespace ie = InferenceEngine;

#include <opencv2/imgcodecs.hpp>

/* Taken from intel samples */
#include "slog.hpp"
#include "yolo.hpp"

string g_ir_path;
string g_cfg_path;
string g_device_type;
string g_image_fpath;

ostream & operator << (ostream &out, const vector<size_t> &c)
{
    out << "[";
    for ( const size_t &s : c )
        out << s << ",";
    out << "]";
    return out;
}

int main(int argc, char **argv)
{
    po::options_description desc("Options for detection app");

    po::options_description_easy_init opt_init = desc.add_options();
    opt_init("help,h", "Print help messages");
    opt_init("ir_path,r", po::value<string>(), "IR path");
    opt_init("cfg_path,c", po::value<string>(), "Net config path");
    opt_init("device,d", po::value<string>()->default_value("CPU"), "Device type: CPU, VPU");
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

    cout << "Input names: " << endl;
    for (const auto &in : net_inputs_info)
    {
        const ie::SizeVector dims = in.second->getTensorDesc().getDims();

        cout << "   " << in.first
             << " / " << dims
             << endl;
    }

    const ie::SizeVector input_dims = net_inputs_info["input_img"]->getTensorDesc().getDims();
    cv::Size input_size(input_dims[3] /* w */, input_dims[2] /* h */);



    ie::OutputsDataMap net_outputs_info(network.getOutputsInfo());

    cout << "Output names: " << endl;
    for (const auto &out : net_outputs_info)
    {
        const ie::SizeVector dims = out.second->getTensorDesc().getDims();

        cout << "   " << out.first
             << " / " << dims
             << endl;
    }



    YOLOConfig main_config(g_cfg_path);

    cv::Mat input_image = cv::imread(g_image_fpath);

    YOLONetwork yolo(main_config, input_size);

    vector<DetectionBox> dets;
    yolo.infer(input_image, dets);

    return EXIT_SUCCESS;
}
