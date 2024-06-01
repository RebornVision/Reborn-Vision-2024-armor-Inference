//
// Created by gx on 24-1-27.
//


#ifndef V8_INFERENCE_H
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>
#include <thread>

#define model_path "../model/mobilenetv3_last_int_all_new/last.xml"
#define score_threshold 0.7
#define nms_threshold 0.3
#define XML_SIZE 416

/*
 * 整体重写的方向为：
 * 1、成为一个独立头文件
 * 2、方便其他文件调用此文件各种函数（安全就不考虑了）
 * 3、分为推理+输出+可视化 ，三部分
 * 4、提供更多的宏定义，便于调整参数（此处就无所谓滥用宏了）
 */

//******************推理**************
//模型加载（懒狗不想写类，全用静态函数了）
struct dataImg{
    float scale;
    cv::Mat blob;
    cv::Mat input;
};
struct Armor{
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> objects_keypoints;
    int class_ids;

};
struct OneArmor{
    float class_scores;
    cv::Rect box;
    cv::Point2f objects_keypoints[4];
    int class_ids;//color
    cv::Mat number_img;
    std::string number;

};

static std::once_flag flag;
static ov::Core core;
static ov::CompiledModel compiled_model;
static ov::InferRequest infer_request;
static ov::Output<const ov::Node> input_port;
dataImg preprocessImage(const cv::Mat& img, cv::Size new_shape=cv::Size( XML_SIZE, XML_SIZE), cv::Scalar color=cv::Scalar(114,114,114)); //图片预处理
std::vector<OneArmor> startInferAndNMS(dataImg data );//开始推理并返回结果

#define V8_INFERENCE_H

#endif //V8_INFERENCE_H
