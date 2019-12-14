#include <chrono>

using namespace std;

#include <ext_list.hpp>
namespace ie = InferenceEngine; 

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "yolo_ov.hpp"

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

YOLO_OpenVINO::YOLO_OpenVINO(std::string cfg_fpath) : 
    CommonYOLO(cfg_fpath)
{
}

bool YOLO_OpenVINO::init(std::string ir_fpath, std::string device_type)
{
    // cout << ie_core.GetVersions(g_device_type) << endl;
    cout << "InferenceEngine: " << ie::GetInferenceEngineVersion() << endl;
    cout << "Loading Inference Engine" << endl;

    if (device_type == "CPU")
    {
        mIeCore.AddExtension(make_shared<ie::Extensions::Cpu::CpuExtensions>(), "CPU");
    }

    string ir_bin_path = fs::path(ir_fpath).replace_extension(".bin").string();

    cout << "Loading network files:\n\t"
         << ir_fpath << "\n\t"
         << ir_bin_path << endl;

    ie::CNNNetReader net_reader;
    net_reader.ReadNetwork(ir_fpath);
    net_reader.ReadWeights(ir_bin_path);
    mNetwork = net_reader.getNetwork();

    cout << "Preparing input blobs" << endl;

    ie::InputsDataMap net_inputs_info(mNetwork.getInputsInfo());
    ie::InputInfo::Ptr &input_data = net_inputs_info.begin()->second;
    input_data->setPrecision(ie::Precision::U8);
    mInputName = net_inputs_info.begin()->first;


    cout << "Loading to device" << endl;
    mExecutableNetwork = mIeCore.LoadNetwork(mNetwork, device_type);

    for ( auto &info : mExecutableNetwork.GetOutputsInfo() )
        mOutputNames.push_back(info.first);

    for ( size_t i = 0; i < mCfg._tile_cnt; i++ )
    {
        ie::InferRequest::Ptr request = mExecutableNetwork.CreateInferRequestPtr();
        mInferRequests.push_back(request);
    }
}

void YOLO_OpenVINO::infer(cv::Mat raw_image, vector<DetectionObject> &detections)
{
    chrono::time_point<chrono::steady_clock> inf_start_time = chrono::steady_clock::now();

    ImageResizeConfig   rsz_cfg;
    initResizeConfig(raw_image, rsz_cfg);

    vector<cv::Mat> input_frames;

    for ( size_t i = 0; i < mInferRequests.size(); i++ )
    {
        ie::InferRequest::Ptr &request = mInferRequests[i];
        ie::Blob::Ptr input_blob = request->GetBlob(mInputName);

        cv::Mat inputFrame;
        rsz_cfg.tile_idx = i;
        resizeForNetwork(raw_image, inputFrame, rsz_cfg);

        input_frames.push_back(inputFrame);
        matU8ToBlob<uint8_t>(inputFrame, input_blob, 0);

        request->StartAsync();
    }

    vector<RawDetectionObject> global_dets;

#define FULL_PROCESSING
    for ( size_t i = 0; i < mInferRequests.size(); i++ )
    {
        ie::InferRequest::Ptr &request = mInferRequests[i];
        request->Wait(ie::IInferRequest::WaitMode::RESULT_READY);

#ifdef FULL_PROCESSING

        for (size_t i_layer = 0; i_layer < mOutputNames.size(); i_layer++)
        {
            const ie::Blob::Ptr output_blob = request->GetBlob(mOutputNames[i_layer]);
            const ie::SizeVector &output_dims = output_blob->getTensorDesc().getDims();

            vector<cv::Point> anchors = get_anchors(i_layer);
            
            const float grid_w = output_dims[3];
            const float grid_h = output_dims[2];
            const size_t chnl_count = output_dims[1];

            get_detections(global_dets, output_blob->buffer().as<void *>(), grid_h, grid_w, chnl_count, anchors, ParsingFormat::CHW);
        }

        rsz_cfg.tile_idx = i;
        postprocessBoxes(global_dets, rsz_cfg);

#endif //FULL_PROCESSING
    }

    filterBoxes(global_dets, detections);

    chrono::duration<double> inf_elapsed = chrono::steady_clock::now() - inf_start_time;
    cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(inf_elapsed).count() << " [ms]" << endl;

}
