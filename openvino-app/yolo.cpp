#include "yolo.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <chrono>
using namespace std;

namespace ie = InferenceEngine; 

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

YOLONetwork::YOLONetwork(YOLOConfig cfg, cv::Size infer_sz) : 
    mCfg(cfg), mInferSize(infer_sz)
{

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


cv::Mat YOLONetwork::get_roi_tile(cv::Mat raw_image, size_t idx)
{
    if (mCfg._tile_cnt == 1)
    {
        return raw_image;
    }
    else if (mCfg._tile_cnt == 2)
    {
        if ( idx == 0 )
        {
            cv::Rect roi_left(cv::Point(0, 0), cv::Point(raw_image.cols/2, raw_image.rows));
            return raw_image(roi_left);
        }
        else if ( idx == 1 )
        {
            cv::Rect roi_right(cv::Point(raw_image.cols/2, 0), cv::Point(raw_image.cols, raw_image.rows));
            return raw_image(roi_right);
        }
    }
}


double sigmoid(double x)
{
    return x / (1.0 + fabs(x));
}

void YOLONetwork::infer(cv::Mat raw_image,
                        ie::ExecutableNetwork &executable_network,
                        vector<DetectionBox> &detections)
{
    chrono::time_point<chrono::steady_clock> inf_start_time = chrono::steady_clock::now();

    vector<ie::InferRequest::Ptr> infer_requests;
    for ( size_t i = 0; i < mCfg._tile_cnt; i++ )
        infer_requests.push_back(executable_network.CreateInferRequestPtr());

    vector<cv::Mat> net_inputs;

    /* Find new size of image */
    cv::Size2f tile_sz;
    vector<cv::Rect> tile_rects;
    if (mCfg._tile_cnt == 1)
    {
        tile_sz = cv::Size2f(raw_image.cols, raw_image.rows);
        tile_rects.push_back( cv::Rect(cv::Point(0, 0), cv::Size(raw_image.cols, raw_image.rows)) );
    }
    else if (mCfg._tile_cnt == 2)
    {
        tile_sz = cv::Size(raw_image.cols/2, raw_image.rows);
        tile_rects.push_back( cv::Rect(cv::Point(0, 0), cv::Size(raw_image.cols/2, raw_image.rows)) );
        tile_rects.push_back( cv::Rect(cv::Point(raw_image.cols/2, 0), cv::Size(raw_image.cols/2, raw_image.rows)) );
    }

    uint32_t top = 0,
             bottom = 0,
             left = 0,
             right = 0;
    uint32_t new_w, new_h;
    if ( (mInferSize.width / tile_sz.width) < (mInferSize.height / tile_sz.height) )
    {
        new_w = mInferSize.width;
        new_h = mInferSize.height / tile_sz.width * mInferSize.width;

        left = right = 0;
        top = (mInferSize.height - new_h) / 2;
        bottom = (mInferSize.height - new_h) - top;
    }
    else
    {
        new_h = mInferSize.height;
        new_w = tile_sz.width / tile_sz.height * mInferSize.height;

        top = bottom = 0;
        left = (mInferSize.width - new_w) / 2;
        right = (mInferSize.width - new_w) - left;
    }

    /* For correction */
    float x_offset = (mInferSize.width - new_w) / 2. / mInferSize.width;
    float y_offset = (mInferSize.height - new_h) / 2. / mInferSize.height;

    float x_scale = new_w / mInferSize.width;
    float y_scale = new_h / mInferSize.height;

    cv::Mat resizedFrame;
    cv::Mat inputFrame;

    vector<cv::Mat> input_frames;

    /* Find new size of image */
    string input_name = executable_network.GetInputsInfo().begin()->first;
    // const ie::SizeVector input_dims = executable_network.GetInputsInfo().begin()->second->getTensorDesc().getDims();
    // cv::Size input_size(input_dims[3] /* w */, input_dims[2] /* h */);

    for ( size_t i = 0; i < infer_requests.size(); i++ )
    {
        ie::InferRequest::Ptr &request = infer_requests[i];

        ie::Blob::Ptr input_blob = request->GetBlob(input_name);

        cv::Mat roi_tile = get_roi_tile(raw_image, i);
        cv::resize(roi_tile, resizedFrame, cv::Size(new_w, new_h));

        // cout << resizedFrame.size() << endl;

        cv::copyMakeBorder(resizedFrame, inputFrame, 
                        top, bottom, left, right, 
                        cv::BORDER_CONSTANT, 
                        cv::Scalar(127, 127, 127));

        // cout << inputFrame.size() << endl;

        input_frames.push_back(inputFrame);
        matU8ToBlob<uint8_t>(inputFrame, input_blob, 0);

        // cout << "Start inference" << endl;
        request->StartAsync();
    }


    vector<vector<RawDetectionBox>> raw_detections(infer_requests.size());
    vector<string> outputNames;

    for ( auto &info : executable_network.GetOutputsInfo() )
    {
        outputNames.push_back(info.first);
    }


    for ( size_t i = 0; i < infer_requests.size(); i++ )
    {
        ie::InferRequest::Ptr &request = infer_requests[i];

        request->Wait(ie::IInferRequest::WaitMode::RESULT_READY);

        vector<RawDetectionBox> &batch_dets = raw_detections[i];

        for (size_t i_layer = 0; i_layer < outputNames.size(); i_layer++)
        {
            const ie::Blob::Ptr output_blob = request->GetBlob(outputNames[i_layer]);
            const ie::SizeVector &output_dims = output_blob->getTensorDesc().getDims();

            vector<cv::Point> anchors = get_anchors(i_layer);

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

            float obj_thresh = 0.5;

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
                            
                            det.box_w = anchors[anc_idx].x * exp(det.box_w) / mInferSize.width;
                            det.box_h = anchors[anc_idx].y * exp(det.box_h) / mInferSize.height;
                            det.box_x = (sigmoid(det.box_x) + w_idx) / grid_w;
                            det.box_y = (sigmoid(det.box_y) + h_idx) / grid_h;

                            for (size_t i_cls = 0; i_cls < class_count; i_cls++)
                            {
                                float class_val = detection[grid_offset + c_stride * ((i_cls + 5) + chnl_offset)];
                                det.cls = sigmoid(class_val) * obj;

                                if ( det.cls < obj_thresh )
                                    continue;
                                
                                det.cls_idx = i_cls;
                                batch_dets.push_back(det);
                            }
                        }
                    }
                }
            }
        }

        cv::Mat net_input_frame = input_frames[i];

        /* Correct each detection */
        for ( RawDetectionBox &det : batch_dets )
        {
            cv::Point tl(
                (det.box_x - det.box_w/2) * mInferSize.width,
                (det.box_y - det.box_h/2) * mInferSize.height);
            cv::Point br(
                (det.box_x + det.box_w/2) * mInferSize.width,
                (det.box_y + det.box_h/2) * mInferSize.height);

            // cout << tl << " / " << br << endl;

            cv::rectangle(net_input_frame, tl, br, cv::Scalar(250, 0, 0), 2);

            DetectionBox px_det;
            px_det.cls = det.cls;
            px_det.cls_idx = det.cls_idx;

            px_det.box_x = (det.box_x - x_offset) / x_scale * tile_rects[i].width;
            px_det.box_y = (det.box_y - y_offset) / y_scale * tile_rects[i].height;

            px_det.box_w = det.box_w / x_scale * tile_rects[i].width;
            px_det.box_h = det.box_h / y_scale * tile_rects[i].height;
         
            px_det.box_x += tile_rects[i].x;
            px_det.box_y += tile_rects[i].y;
            
            detections.push_back(px_det);
        }

        cv::imshow("net", net_input_frame);
        cv::waitKey(0);
    }

    chrono::duration<double> inf_elapsed = chrono::steady_clock::now() - inf_start_time;
    cout << "Inference time: " << chrono::duration_cast<chrono::milliseconds>(inf_elapsed).count() << " [ms]" << endl;

}
