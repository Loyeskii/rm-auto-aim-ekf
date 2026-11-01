#include "camera.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace rm {

Undistorter::Undistorter(const CameraIntrinsic& calib, cv::Size actual) {
    if (calib.K.empty() || calib.calib_width <= 0 || calib.calib_height <= 0) {
        return;  // 参数不全，保持 invalid 状态；apply() 会原样返回
    }

    // ---- 1) 内参按实际分辨率缩放 ----
    // 素材的内参是按 1358x1058 标定的，而图片是 960x720（非等比缩放来的），
    // 若不做这一步，fx/fy/cx/cy 全部错位，解算结果必然偏移。
    const double sx = static_cast<double>(actual.width)  / static_cast<double>(calib.calib_width);
    const double sy = static_cast<double>(actual.height) / static_cast<double>(calib.calib_height);

    K_ = calib.K.clone();
    K_.convertTo(K_, CV_64F);
    K_.at<double>(0, 0) *= sx;   // fx
    K_.at<double>(0, 2) *= sx;   // cx
    K_.at<double>(1, 1) *= sy;   // fy
    K_.at<double>(1, 2) *= sy;   // cy
    // K(0,1) 为像素轴倾斜，通常为 0，不处理

    cv::Mat D = calib.D.clone();
    D.convertTo(D, CV_64F);

    // ---- 2) 预计算重映射表 ----
    // R 传空矩阵（等价于单位阵），newCameraMatrix 传 K_：
    // 这样"去畸变后图像"的坐标系与 K_ 严格对应。
    cv::initUndistortRectifyMap(K_, D, cv::Mat(), K_, actual,
                                CV_32FC1, map1_, map2_);
}

cv::Mat Undistorter::apply(const cv::Mat& src) const {
    if (map1_.empty() || src.size() != map1_.size()) {
        return src.clone();   // 尺寸不匹配就原样返回，避免 remap 崩溃
    }
    cv::Mat dst;
    cv::remap(src, dst, map1_, map2_, cv::INTER_LINEAR);
    return dst;
}

}  // namespace rm
