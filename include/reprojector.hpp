#pragma once

// 由估计出的整车状态反推四块装甲板的位置，并投影回图像。
//
// 画的是「灯条外角矩形」——和检测网络、PnP 物方点是同一套尺寸，
// 因此黄框（当前观测到的那块）可以直接和红框（检测结果）重合比对；
// 如果画成整块板的轮廓（板高 125），就会因为高度不同而永远对不上。
//
// 这一步是观测模型的自检：h(x, phi) 里的相位、r、za 一旦写错，
// 四块板会整体转到错误的方向上，比看曲线更容易发现。

#include "config.hpp"

#include <opencv2/core.hpp>

namespace rm {

class Reprojector {
public:
    // 画出估计出的四块板。phase_idx 是当前观测到的板号（画成黄色），其余画绿色。
    // K：去畸变后内参；R_cw：世界→相机 旋转
    static void drawEstimatedArmors(cv::Mat& img, const cv::Mat& K,
                                    const cv::Matx33d& R_cw, const cv::Mat& x,
                                    bool is_large, int phase_idx, const Config& cfg);

    // 右上角状态面板：实时显示 9 个状态量
    static void drawStatePanel(cv::Mat& img, const cv::Mat& x, const cv::Mat& P,
                               int frame, bool tracking, bool switched, double nis);
};

}  // namespace rm
