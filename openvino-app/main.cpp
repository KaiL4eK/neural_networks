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

static ie::Blob::Ptr mat_to_blob(const cv::Mat &image)
{
    ie::TensorDesc tensor(ie::Precision::U8, {1, (size_t)image.channels(), (size_t)image.size().height, (size_t)image.size().width}, ie::Layout::NHWC);
    return ie::make_shared_blob<uint8_t>(tensor, image.data);
}

// /**
//  * @brief Wraps data stored inside of a passed cv::Mat object by new Blob pointer.
//  * @note: No memory allocation is happened. The blob just points to already existing
//  *        cv::Mat data.
//  * @param mat - given cv::Mat object with an image data.
//  * @return resulting Blob pointer.
//  */
// static InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat) {
//     size_t channels = mat.channels();
//     size_t height = mat.size().height;
//     size_t width = mat.size().width;

//     size_t strideH = mat.step.buf[0];
//     size_t strideW = mat.step.buf[1];

//     bool is_dense =
//             strideW == channels &&
//             strideH == channels * width;

//     if (!is_dense) THROW_IE_EXCEPTION
//                 << "Doesn't support conversion from not dense cv::Mat";

//     InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
//                                       {1, channels, height, width},
//                                       InferenceEngine::Layout::NHWC);

//     return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
// }

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat &orig_image, InferenceEngine::Blob::Ptr &blob, int batchIndex = 0)
{
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    T *blob_data = blob->buffer().as<T *>();

    if (static_cast<int>(width) != orig_image.size().width ||
        static_cast<int>(height) != orig_image.size().height)
    {
        throw invalid_argument("Invalid size!");
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < height; h++)
        {
            for (size_t w = 0; w < width; w++)
            {
                blob_data[batchOffset + c * width * height + h * width + w] =
                    orig_image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

double sigmoid(double x)
{
    return x / (1.0 + fabs(x));
}

void softmax(float *classes, size_t sz)
{
    float sum = 0;

    for (size_t i = 0; i < sz; i++)
    {
        classes[i] = exp(classes[i]);
        sum += classes[i];
    }

    for (size_t i = 0; i < sz; i++)
    {
        classes[i] = classes[i] / sum;
    }
}

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

    if (main_config._tile_cnt != network.getBatchSize())
    {
        cout << "Resizing batch size to " << main_config._tile_cnt << endl;
        network.setBatchSize(main_config._tile_cnt);
    }

    cv::Mat input_image = cv::imread(g_image_fpath);

    YOLONetwork yolo(main_config, input_size);

    cout << "Softmax test: ";
    float sample[4] = {1, 4, 3, 4};
    softmax(sample, 4);
    for (float &val : sample)
        cout << val << ", ";
    cout << endl;

    vector<cv::Mat> net_inputs;
    vector<ie::Blob::Ptr> net_blobs;

    // cout << "Loading to device" << endl;
    ie::ExecutableNetwork executable_network = ie_core.LoadNetwork(network, g_device_type);
    ie::InferRequest infer_request = executable_network.CreateInferRequest();

    ie::Blob::Ptr input_blob = infer_request.GetBlob(input_data->name());

    cout << "Input blob info:" << endl
         << "  " << input_blob->getTensorDesc().getPrecision() << endl
         << "  " << input_blob->getTensorDesc().getDims() << endl;

    size_t batchSize = network.getBatchSize();
    cout << "Batch size is " << batchSize << endl;

    chrono::time_point<chrono::steady_clock> start_time = chrono::steady_clock::now();

    yolo.get_inputs(input_image, net_inputs);

    net_blobs.clear();
    for (size_t i = 0; i < net_inputs.size(); i++)
    {
        matU8ToBlob<uint8_t>(net_inputs[i], input_blob, i);

        // ie::Blob::Ptr ptr = mat_to_blob(input);

        // net_blobs.push_back(ptr);
    }

    chrono::duration<double> elapsed = chrono::steady_clock::now() - start_time;
    cout << "Convertion time: " << chrono::duration_cast<chrono::milliseconds>(elapsed).count() << " [ms]" << endl;

    cout << "Start inference" << endl;
    chrono::time_point<chrono::steady_clock> inf_start_time = chrono::steady_clock::now();

    infer_request.Infer();

    chrono::duration<double> inf_elapsed = chrono::steady_clock::now() - inf_start_time;
    cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(inf_elapsed).count() << " [ms]" << endl;

    vector<vector<RawDetectionBox>> raw_detections(batchSize);

    chrono::time_point<chrono::steady_clock> proc_start_time = chrono::steady_clock::now();

    for (size_t i_layer = 0; i_layer < outputNames.size(); i_layer++)
    {
        const ie::Blob::Ptr output_blob = infer_request.GetBlob(outputNames[i_layer]);
        const ie::SizeVector &output_dims = output_blob->getTensorDesc().getDims();

        cout << "Output blob info:" << endl
             << "  " << outputNames[i_layer] << endl
             << "  " << output_blob->getTensorDesc().getPrecision() << endl
             << "  " << output_dims << endl;

        vector<cv::Point> anchors = yolo.get_anchors(i_layer);

        for (const cv::Point &anchor : anchors)
            cout << anchor << endl;

        float grid_w = output_dims[3];
        float grid_h = output_dims[2];
        size_t chnl_count = output_dims[1];

        const float *detection = static_cast<ie::PrecisionTrait<ie::Precision::FP32>::value_type *>(output_blob->buffer());
        size_t b_stride = (output_dims[1] * output_dims[2] * output_dims[3]);
        size_t c_stride = (output_dims[2] * output_dims[3]);
        size_t h_stride = output_dims[3];
        size_t c_idx;

        size_t class_count = output_dims[1] / anchors.size() - 5;
        size_t box_count = class_count + 5;
        cout << "Class count: " << class_count << endl;

        float obj_thresh = 0.5;
        RawDetectionBox det;
        float classes[class_count];

        for (size_t b_idx = 0; b_idx < output_dims[0]; b_idx++)
        {
            for (size_t h_idx = 0; h_idx < output_dims[2]; h_idx++)
            {
                for (size_t w_idx = 0; w_idx < output_dims[3]; w_idx++)
                {
                    size_t grid_offset = b_idx * b_stride + h_idx * h_stride + w_idx;

                    for (size_t anc_idx = 0; anc_idx < anchors.size(); anc_idx++)
                    {
                        RawDetectionBox det;
                        size_t chnl_offset = anc_idx * box_count;

                        // size_t box_idx_x = 0;
                        // size_t box_idx_y = 1;
                        // size_t box_idx_w = 2;
                        // size_t box_idx_h = 3;
                        // size_t obj_idx = 4;
                        // size_t cls_idx = 5;

                        float obj = detection[grid_offset + c_stride * (4 + chnl_offset)];
                        obj = sigmoid(obj);
                        if (obj < obj_thresh)
                            continue;

                        

                        det.box_x = detection[grid_offset + c_stride * (0 + chnl_offset)];
                        det.box_y = detection[grid_offset + c_stride * (1 + chnl_offset)];
                        det.box_w = detection[grid_offset + c_stride * (2 + chnl_offset)];
                        det.box_h = detection[grid_offset + c_stride * (3 + chnl_offset)];

                        for (size_t i_cls = 0; i_cls < class_count; i_cls++)
                        {
                            classes[i_cls] = detection[grid_offset + c_stride * ((i_cls + 5) + chnl_offset)];
                        }

                        softmax(classes, class_count);

                        for (size_t i_cls = 0; i_cls < class_count; i_cls++)
                        {
                            classes[i_cls] *= obj;

                            if (classes[i_cls] < obj_thresh)
                                classes[i_cls] = 0;
                        }

                        det.box_w = anchors[anc_idx].x * exp(det.box_w) / input_size.width;
                        det.box_h = anchors[anc_idx].y * exp(det.box_h) / input_size.height;

                        det.box_x = (sigmoid(det.box_x) + w_idx) / grid_w;
                        det.box_y = (sigmoid(det.box_y) + h_idx) / grid_h;

                        det.cls = vector<float>(classes, classes + class_count);

                        raw_detections[b_idx].push_back(det);
                    }
                }
            }
        }
    }

    chrono::duration<double> proc_elapsed = chrono::steady_clock::now() - proc_start_time;
    cout << "Processing time: " << chrono::duration_cast<chrono::milliseconds>(proc_elapsed).count() << " [ms]" << endl;

    for ( size_t i = 0; i < net_inputs.size(); i++ )
    {
        cv::Mat rndr_frame = net_inputs[i];
        for (RawDetectionBox &det : raw_detections[i])
        {
            cv::Point tl(
                (det.box_x - det.box_w/2) * rndr_frame.cols,
                (det.box_y - det.box_h/2) * rndr_frame.rows);
            cv::Point br(
                (det.box_x + det.box_w/2) * rndr_frame.cols,
                (det.box_y + det.box_h/2) * rndr_frame.rows);

            cv::rectangle(rndr_frame, tl, br, cv::Scalar(250, 0, 0), 2);
        }

        cv::imshow("Boxes", rndr_frame);
        cv::waitKey(0);
    }



    return EXIT_SUCCESS;
}
