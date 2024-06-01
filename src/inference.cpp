//
// Created by gx on 24-1-27.
//
#include "../include/inference.hpp"
#include <ie_core.hpp>
static void using_once();
static void Initialize()
{
    std::cout<<"initialize"<<std::endl;
    compiled_model = core.compile_model(model_path,"CPU");
//    core.set_property({ { CONFIG_KEY(CPU_BIND_THREAD), "NO" } });
//    compiled_model= core.compile_model(model_path, "CPU",ov::inference_num_threads(16));

    infer_request = compiled_model.create_infer_request();
    input_port = compiled_model.input();

}

static void using_once()
{
    std::call_once(flag, Initialize);

}
//我们发现神经网络推理的点有时顺序会有误，推断是数据集问题，此函数用于纠正点的顺序
static void sort_keypoints(cv::Point2f keypoints[4]) {
    // Sort points based on their y-coordinates (ascending)
    std::sort(keypoints, keypoints + 4, [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });

    // Top points will be the first two, bottom points will be the last two
    cv::Point top_points[2] = { keypoints[0], keypoints[1] };
    cv::Point bottom_points[2] = { keypoints[2], keypoints[3] };

    // Sort the top points by their x-coordinates to distinguish left and right
    std::sort(top_points, top_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Sort the bottom points by their x-coordinates to distinguish left and right
    std::sort(bottom_points, bottom_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Assign sorted points back to the keypoints array
    keypoints[0] = top_points[0];     // top-left
    keypoints[1] = bottom_points[0];  // bottom-left
    keypoints[2] = bottom_points[1];  // bottom-right
    keypoints[3] = top_points[1];     // top-right
}
dataImg preprocessImage(const cv::Mat& img, cv::Size new_shape, cv::Scalar color)
{
    // Get current shape [height, width]
    cv::Size shape = img.size();

    // Scale ratio (new / old)
    double r = std::min((double)new_shape.height / shape.height, (double)new_shape.width / shape.width);

    // Compute padding
    cv::Size new_unpad = cv::Size(int(round(shape.width * r)), int(round(shape.height * r)));
    int dw = (new_shape.width - new_unpad.width) ;
    int dh = (new_shape.height - new_unpad.height) ;

    // Resize image if necessary
    cv::Mat resized_img;
    if (shape != new_unpad) {
        cv::resize(img, resized_img, new_unpad, 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = img;
    }

    // Add border/padding
    int top = 0;
    int bottom = dh;
    int left = 0 ;
    int right = dw;
    cv::Mat bordered_img;
    cv::copyMakeBorder(resized_img, bordered_img, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    dataImg data;
    float scale = 1.0/(std::min( XML_SIZE*1.0/ img.rows,  XML_SIZE*1.0 / img.cols));
    cv :: Mat blob = cv::dnn::blobFromImage(bordered_img, 1.0 / 255.0, cv::Size( XML_SIZE, XML_SIZE), cv::Scalar(), true); // 图像像素归一化
    data.blob = blob;
    data.input = img;
    data.scale = scale;
    return data;
}
#if 0           // openMP
dataImg preprocessImage(cv::Mat imgInput) {
    int col = imgInput.cols;
    int row = imgInput.rows;
    int _max = std::max(col, row);

    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    imgInput.copyTo(result(Rect(0, 0, col, row)));

    dataImg data;
    float scale;

#pragma omp parallel sections
    {
#pragma omp section
        {
            // 计算图像的缩放比例
            scale = result.size[0] / 640;
        }

#pragma omp section
        {
            Mat blob = blobFromImage(result, 1.0 / 255.0, Size(640,640), Scalar(), true); // 图像像素归一化
            data.blob = blob;
        }
    }

    data.scale = scale;
    data.input = imgInput;

    return data;
}
#endif
std::vector<OneArmor> startInferAndNMS(dataImg img_data ){
    using_once();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), img_data.blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);
    // -------- Start inference --------
    infer_request.infer();
   // std::cout<<"Run into startInferAndNMS.." << std::endl;
    // -------- Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();

    // -------- Postprocess the result --------
    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,14]
    std::vector<OneArmor> qualifiedArmors;

    for (int cls=4 ; cls < 6; ++cls) {
        Armor SingleData;
        for (int i = 0; i < output_buffer.rows; i++) {
            float class_score = output_buffer.at<float>(i, cls);
            //保证当前对应的板子信息匹配
            float max_class_score = 0.0;
            for (int j = 4; j < 6; j++) {
                if(max_class_score < output_buffer.at<float>(i, j)){
                    max_class_score = output_buffer.at<float>(i, j);
                }
            }
            if (class_score != max_class_score){
                continue;
            }

            if (class_score > score_threshold) {

                SingleData.class_scores.push_back(class_score);

                float cx = output_buffer.at<float>(i, 0);
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2);
                float h = output_buffer.at<float>(i, 3);

                // Get the box
                //由于yolov8-pose推理的框不一定满足传统识别取灯条角点所需，可以适当扩大ROI区域
                int left = int((cx - 0.5 * w-2) * img_data.scale);
                int top = int((cy - 0.5 * h-2) * img_data.scale);
                int width = int(w *1.2* img_data.scale);
                int height = int(h *1.2* img_data.scale);

                // Get the keypoints
                std::vector<float> keypoints;
                cv::Mat kpts = output_buffer.row(i).colRange(6,14 );
                for (int i = 0; i < 4; i++) {
                    float x = kpts.at<float>(0, i * 2 + 0) * img_data.scale;
                    float y = kpts.at<float>(0, i * 2 + 1) * img_data.scale;

                    keypoints.push_back(x);
                    keypoints.push_back(y);

                }

                SingleData.boxes.push_back(cv::Rect(left, top, width, height));
                SingleData.objects_keypoints.push_back(keypoints);

            }

        }
        SingleData.class_ids = cls - 4;
       //NMS处理
        std::vector<int> indices;
        cv::dnn::NMSBoxes(SingleData.boxes, SingleData.class_scores, score_threshold, nms_threshold, indices);

      //  std::cout << "indices: " << indices.size() << std::endl;
        for(auto i:indices)
        {
            OneArmor armor;

            armor.box = SingleData.boxes[i];
            armor.class_scores = SingleData.class_scores[i];
            armor.class_ids = SingleData.class_ids;

            for (int j = 0; j < 4; j++) {
                int x = std::clamp(int(SingleData.objects_keypoints[i][j * 2 + 0]), 0, img_data.input.cols);
                int y = std::clamp(int(SingleData.objects_keypoints[i][j * 2 + 1]), 0, img_data.input.rows);
                armor.objects_keypoints[j] = cv::Point(x, y);


            }
            sort_keypoints(armor.objects_keypoints);
            qualifiedArmors.push_back(armor);
        }
    }
    return qualifiedArmors;
}

