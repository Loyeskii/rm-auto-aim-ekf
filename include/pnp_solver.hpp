#pragma once

// 位置解算（PnP）。
// 对应理论：Perspective-n-Point 位姿估计、共面四点解法(IPPE)、
//           旋转的向量表示(罗德里格斯)、重投影误差。
//
// 核心：已知装甲板真实尺寸（物方 4 点）+ 图像上检测到的 4 个角点（像素 2D 点），
//       求相机坐标系下装甲板的平移 tvec 和旋转 rvec。

#include "config.hpp"
#include "types.hpp"
#include <opencv2/core.hpp>

namespace rm {

class PnPSolver {
public:
    explicit PnPSolver(const Config& cfg) : cfg_(cfg) {}

    // 成功时写入 armor.rvec / armor.tvec / armor.reproj_err，并置 solved=true
    bool solve(Armor& armor, const cv::Mat& K) const;

private:
    Config cfg_;
};

}  // namespace rm
