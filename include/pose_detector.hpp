#pragma once

// 装甲板检测（神经网络方案的核心模块）。
//
// 对应理论：单阶段目标检测（YOLOv8-pose）、张量解码、非极大值抑制(NMS)、
//           关键点回归、图像预处理（resize / 归一化 / 通道序）。
//
// 该网络一次前向同时给出三样东西：
//   1) 类别分数  → 9 类，实测 class N 对应号码 N
//   2) 四个关键点 → 装甲板的四个角（顺序见 .cpp 里的说明）
//   3) 检测框     → 实测该模型输出的宽高恒为 0，**不可用**，
//                   因此框由四个关键点自己算，而不是读网络的 cx,cy,w,h

#include "config.hpp"
#include "types.hpp"
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace rm {

class PoseDetector {
public:
    explicit PoseDetector(const Config& cfg);

    bool ready() const { return !net_.empty(); }
    const std::string& loadError() const { return err_; }

    // 输入原始 BGR 图（已去畸变），输出候选装甲板
    std::vector<Armor> detect(const cv::Mat& bgr);

private:
    // 网络单次前向的原始输出：[1, 21, 6300]
    //   21 = 4(box) + 9(cls) + 8(4 关键点 × xy)
    //   6300 = 60×80 + 30×40 + 15×20（stride 8/16/32 三尺度）
    cv::Mat forward(const cv::Mat& bgr);

    Config cfg_;
    cv::dnn::Net net_;
    std::string err_;
};

}  // namespace rm
