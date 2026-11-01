#pragma once

// 全项目共用的数据结构定义（神经网络方案）。
//
// 与方案 A（传统视觉）的区别：
//   A 从"灯条"出发，先有 LightBar 再配对成 Armor；
//   B 由检测网络直接给出"一块装甲板 + 四个角点"，没有灯条这个中间概念，
//   所以这里不再定义 LightBar。

#include <opencv2/core.hpp>
#include <string>

namespace rm {

// 装甲板颜色。神经网络方案不区分红蓝（网络输出里没有颜色类别），
// 保留该字段只为让可视化代码与方案 A 共用。
enum class ArmorColor { RED = 0, BLUE = 1, UNKNOWN = 2 };

// 相机内参 + 畸变系数
struct CameraIntrinsic {
    cv::Mat K;             // 3x3 (CV_64F)
    cv::Mat D;             // 1x5 [k1,k2,p1,p2,k3] (CV_64F)
    int calib_width = 0;
    int calib_height = 0;
};

// 一块装甲板：检测 + 号码 + 位姿的完整结果
struct Armor {
    // 四个角点，顺序固定：左上、右上、右下、左下。
    // 注意：检测网络输出的关键点顺序是「左上、左下、右下、右上」，
    //       PoseDetector 解码时会重排成这里的顺序。
    cv::Point2f corners[4];

    ArmorColor color = ArmorColor::UNKNOWN;
    bool is_large = false;

    int number = -1;           // 类别索引（= labels.txt 的行号）
    std::string number_label;  // 号码文本
    float digit_conf = 0.f;    // 类别置信度

    cv::Vec3d rvec{0, 0, 0};
    cv::Vec3d tvec{0, 0, 0};
    bool solved = false;
    double reproj_err = -1.0;  // 重投影误差（像素）
};

}  // namespace rm
