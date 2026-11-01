#pragma once

// 相机模型与畸变校正。
// 对应理论：针孔成像模型、内参矩阵、径向/切向畸变、图像重映射(remap)。
//
// 关键设计：去畸变时把 newCameraMatrix 取成"缩放后的内参 K"本身，
// 于是去畸变之后的图像仍然用这个 K 描述 —— 后面 solvePnP 直接用 K 即可，
// 不需要再做任何坐标换算。

#include "types.hpp"
#include <opencv2/core.hpp>

namespace rm {

class Undistorter {
public:
    Undistorter() = default;

    // calib: 标定得到的内参；actual: 实际要处理的图像尺寸
    Undistorter(const CameraIntrinsic& calib, cv::Size actual);

    // 对一帧做去畸变
    cv::Mat apply(const cv::Mat& src) const;

    // 去畸变后图像对应的内参（已按实际分辨率缩放）
    const cv::Mat& K() const { return K_; }

    bool valid() const { return !map1_.empty(); }

private:
    cv::Mat K_;              // 缩放后的内参 3x3 CV_64F
    cv::Mat map1_, map2_;    // 预计算的重映射表
};

}  // namespace rm
