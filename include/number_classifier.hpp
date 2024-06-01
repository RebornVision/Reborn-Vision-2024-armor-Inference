// 一个rv改的数字推理代码，数字模型可直接替换rv的mlp.onnx
#ifndef ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_
#define ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "inference.hpp"

#define number_classifier_model_path_ "../model/number_classifier.onnx"

static std::once_flag flag_;
static cv::dnn::Net net_;
static std::vector<std::string> class_names_ = {
    "1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"};

void extractNumbers(const cv::Mat &src, std::vector<OneArmor> &armors);

std::vector<OneArmor> classify(cv::Mat src, std::vector<OneArmor> &armors);

#endif // ARMOR_DETECTOR__NUMBER_CLASSIFIER_HPP_