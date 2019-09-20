#include "yolo_ov.hpp"

using namespace std;

#include <ext_list.hpp>
namespace ie = InferenceEngine; 

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

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

            const float *detection = static_cast<ie::PrecisionTrait<ie::Precision::FP32>::value_type *>(output_blob->buffer());
            const size_t b_stride = (output_dims[1] * output_dims[2] * output_dims[3]);
            const size_t c_stride = (output_dims[2] * output_dims[3]);
            const size_t h_stride = output_dims[3];
            size_t c_idx;

            const size_t class_count = output_dims[1] / anchors.size() - 5;
            const size_t box_count = class_count + 5;

            float obj_thresh = mCfg._objectness_thresh;

            for (size_t b_idx = 0; b_idx < output_dims[0]; b_idx++)
            {
                for (size_t h_idx = 0; h_idx < output_dims[2]; h_idx++)
                {
                    for (size_t w_idx = 0; w_idx < output_dims[3]; w_idx++)
                    {
                        size_t grid_offset = b_idx * b_stride + h_idx * h_stride + w_idx;

                        for (size_t anc_idx = 0; anc_idx < anchors.size(); anc_idx++)
                        {
                            RawDetectionObject det;
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

                            det.x = detection[grid_offset + c_stride * (0 + chnl_offset)];
                            det.y = detection[grid_offset + c_stride * (1 + chnl_offset)];
                            det.w = detection[grid_offset + c_stride * (2 + chnl_offset)];
                            det.h = detection[grid_offset + c_stride * (3 + chnl_offset)];
                            
                            det.w = anchors[anc_idx].x * exp(det.w) / mCfg._infer_sz.width;
                            det.h = anchors[anc_idx].y * exp(det.h) / mCfg._infer_sz.height;
                            det.x = (sigmoid(det.x) + w_idx) / grid_w;
                            det.y = (sigmoid(det.y) + h_idx) / grid_h;

                            for (size_t i_cls = 0; i_cls < class_count; i_cls++)
                            {
                                float class_val = detection[grid_offset + c_stride * ((i_cls + 5) + chnl_offset)];
                                det.conf = sigmoid(class_val) * obj;

                                if ( det.conf < obj_thresh )
                                    continue;
                                
                                det.cls_idx = i_cls;
                                global_dets.push_back(det);
                            }
                        }
                    }
                }
            }
        }

        rsz_cfg.tile_idx = i;
        postprocessBoxes(global_dets, rsz_cfg);

#endif //FULL_PROCESSING
    }

    filterBoxes(global_dets, detections);

    chrono::duration<double> inf_elapsed = chrono::steady_clock::now() - inf_start_time;
    cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(inf_elapsed).count() << " [ms]" << endl;

}
