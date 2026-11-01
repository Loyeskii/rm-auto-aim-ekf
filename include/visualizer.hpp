#pragma once

// 结果可视化：画装甲板框、数字标签、解算坐标、三维坐标轴。
// 对应理论：点在图像上的绘制、三维点重投影、视频帧写入。

#include "types.hpp"
#include <opencv2/core.hpp>
#include <string>

namespace rm {

class Visualizer {
public:
    // 在图上绘制一块装甲板的全部标注
    // K 为去畸变后图像的内参，用于画三维坐标轴
    static void drawArmor(cv::Mat& img, const Armor& armor, const cv::Mat& K);

    // 左上角状态栏
    static void drawHud(cv::Mat& img, int frame_idx, int armor_count);
};

}  // namespace rm
