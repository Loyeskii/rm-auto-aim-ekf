#pragma once

// 九维扩展卡尔曼滤波：整车状态估计。
//
// 状态（顺序与代码一致）：
//   0 xc     车中心 x        [m]
//   1 v_xc   中心 x 速度     [m/s]
//   2 yc     车中心 y        [m]
//   3 v_yc   中心 y 速度     [m/s]
//   4 za     当前参考板高度  [m]
//   5 v_za   高度变化率      [m/s]
//   6 yaw    参考板朝向      [rad]   （指向"板→车中心"方向）
//   7 v_yaw  自转角速度      [rad/s]
//   8 r      板到中心距离    [m]
//
// 量测（4 维，世界系）：
//   z = [x_a, y_a, z_a, yaw_a]
//
// 观测方程（注意减号：yaw 指向"板→中心"，所以从中心找板要减半径）：
//   x_a = xc - r*cos(yaw + phi)
//   y_a = yc - r*sin(yaw + phi)
//   z_a = za
//   yaw_a = yaw + phi
//
// 其中 phi = j * 2pi/N 是"第 j 块板"相对参考板的相位。
// 把相位显式建模进观测，换板时只需改 phi，状态本身连续——不必硬掰 yaw。

#include "config.hpp"

#include <opencv2/core.hpp>

namespace rm {

constexpr int kStateDim = 9;
constexpr int kMeasDim = 4;

class Ekf {
public:
    explicit Ekf(const Config& cfg);

    // 用第一块看到的板初始化：r 取默认值反推车中心，速度置零
    void init(const cv::Vec3d& armor_pos, double armor_yaw);

    // 预测一步（dt 秒），同时传播协方差
    void predict(double dt);

    // 用一帧量测更新。量测被新息门控拒绝时返回 false（状态不变）
    // phase：本次观测的是第几块板相对参考板的相位（弧度）
    bool update(const cv::Vec4d& z, const cv::Mat& R, double nis_gate, double* nis,
                double phase = 0.0);

    // 由状态算出第 j 块板（相位 phase）的预测位置，匹配时用来求预测位置
    cv::Vec3d predictedArmorPos(double phase = 0.0) const;

    // 位置新息的马氏距离平方：用预测协方差 + 量测协方差一起归一化。
    // 起手时 P 大、门就松；收敛后 P 小、门就紧，比固定距离阈值合理。
    double mahalanobisPos(const cv::Vec3d& p_meas, const cv::Mat& R_pos,
                          double phase = 0.0) const;

    const cv::Mat& x() const { return x_; }
    const cv::Mat& P() const { return P_; }
    double at(int i) const { return x_.at<double>(i); }
    double var(int i) const { return P_.at<double>(i, i); }

    // 跳变处理用：直接改写某几个分量，不动协方差
    void setStateValue(int idx, double v) { x_.at<double>(idx) = v; }

    // 跳变帧用：放大某个分量的方差。
    // 换到另一块板上，"对半径/高度的旧把握"不再成立，必须临时放宽。
    void inflateVariance(int idx, double factor);

    // 发散保护：位置与三个速度按当前量测重建
    void resetPosVel(const cv::Vec3d& armor_pos);

    cv::Matx33d rotationFromYaw(double yaw) const;

private:
    cv::Mat jacobianH(double phase) const;
    cv::Mat processNoise(double dt) const;

    Config cfg_;
    cv::Mat x_;  // 9x1
    cv::Mat P_;  // 9x9
};

}  // namespace rm
