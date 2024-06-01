
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

// STL
#include <vector>

#include "../include/inference.hpp"
#include "../include/number_classifier.hpp"

static void using_once();
static void Initialize()
{
    net_ =  cv::dnn::readNetFromONNX(number_classifier_model_path_);
}

static void using_once()
{
    std::call_once(flag_, Initialize);

}
//static cv::Mat perform_opening(const cv::Mat& input_image, int kernel_size) {
//    // Check if the input image is valid
//    if (input_image.empty()) {
//        std::cerr << "Input image is empty!" << std::endl;
//        return cv::Mat();
//    }
//
//    // Create a structuring element (kernel) for the morphological operation
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
//
//    // Perform the opening operation
//    cv::Mat opened_image;
//    cv::morphologyEx(input_image, opened_image, cv::MORPH_OPEN, kernel);
//
//    return opened_image;
//}

void extractNumbers(const cv::Mat & src, std::vector<OneArmor> & armors)
{
    // static int num = 0;
    // Light length in image
    const int light_length = 12;
    // Image size after warp
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 54;
    // Number ROI size
    const cv::Size roi_size(20, 28);

    for (auto & armor : armors) {
        // Warp perspective transform
        cv::Point2f lights_vertices[4] = {armor.objects_keypoints[1], armor.objects_keypoints[0], armor.objects_keypoints[3], armor.objects_keypoints[2]};

        const int top_light_y = (warp_height - light_length) / 2 - 1;
        const int bottom_light_y = top_light_y + light_length;
        const int warp_width = small_armor_width;//全按小装甲板处理
        cv::Point2f target_vertices[4] = {
                cv::Point(0, bottom_light_y),
                cv::Point(0, top_light_y),
                cv::Point(warp_width - 1, top_light_y),
                cv::Point(warp_width - 1, bottom_light_y),
        };
        cv::Mat number_image;
        auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
        cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

        // Get ROI
        number_image =
                number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

        // Binarize
        std::vector<cv::Mat> channels(3);
        cv::split(number_image, channels);

        cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
//保存数字图案20*28
        // cv::imwrite("/home/gx/rm_classifier_training-main/补_/"+std::to_string(num++)+".jpg", number_image);
        cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        // cv::Mat number_img_ = perform_opening(number_image,2);

        armor.number_img = number_image;

    }
}

std::vector<OneArmor> classify(cv::Mat src,std::vector<OneArmor> & armors)
{
    extractNumbers(src,armors);

    using_once();
    std::vector<OneArmor> armors_data_;

    for (auto & armor : armors) {
        cv::Mat image = armor.number_img.clone();

        // Normalize
        image = image / 255.0;

        // Create blob from image
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob);

        // Set the input blob for the neural network
        net_.setInput(blob);
        // Forward pass the image blob through the model
        cv::Mat outputs = net_.forward();

        // Do softmax
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob);
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
        softmax_prob /= sum;

        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
        int label_id = class_id_point.x;

        armor.number = class_names_[label_id];

        for(auto i:armors){
            if(i.number != "negative"){
                armors_data_.push_back(i);
            }
        }

    }
    return armors_data_;
}


