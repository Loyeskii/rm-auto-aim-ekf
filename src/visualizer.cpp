#include "visualizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <vector>

namespace rm {

void Visualizer::drawArmor(cv::Mat& img, const Armor& armor, const cv::Mat& K) {
    // 本项目不区分红蓝方（神经网络方案本身也不输出颜色），
    // 统一用红框标记装甲板位置——深色背景下红色最醒目。
    const cv::Scalar color = cv::Scalar(0, 0, 255);  // BGR 红

    // ---- 1) 四边形框 ----
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (int i = 0; i < 4; ++i) {
        poly.emplace_back(cvRound(armor.corners[i].x), cvRound(armor.corners[i].y));
    }
    cv::polylines(img, poly, /*isClosed=*/true, color, 3, cv::LINE_AA);

    char buf[160];

    // ---- 2) 数字标签（贴在框上方，空间不够时改到下方）----
    const std::string label = armor.number_label.empty() ? "?" : armor.number_label;
    std::snprintf(buf, sizeof(buf), "%s  %.2f", label.c_str(), armor.digit_conf);
    cv::Point org = poly[0] + cv::Point(0, -10);
    if (org.y < 16) org.y = poly[3].y + 24;
    cv::putText(img, buf, org, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);

    if (!armor.solved) return;

    // ---- 3) 解算出的坐标 ----
    std::snprintf(buf, sizeof(buf), "x=%.3f y=%.3f z=%.3f m",
                  armor.tvec[0], armor.tvec[1], armor.tvec[2]);
    cv::putText(img, buf, poly[3] + cv::Point(0, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    // ---- 4) 三维坐标轴：直观展示朝向（X 红 / Y 绿 / Z 蓝）----
    // 这就是 PnP 解出的 rvec/tvec 的几何含义：把板坐标系画到图像上。
    cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        rvec.at<double>(i) = armor.rvec[i];
        tvec.at<double>(i) = armor.tvec[i];
    }
    cv::drawFrameAxes(img, K, cv::Mat(), rvec, tvec,
                      /*length=*/0.08f, /*thickness=*/2);
}

void Visualizer::drawHud(cv::Mat& img, int frame_idx, int armor_count) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "frame %d | armor %d", frame_idx, armor_count);
    cv::putText(img, buf, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
}

}  // namespace rm
