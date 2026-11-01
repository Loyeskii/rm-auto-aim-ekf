#pragma once

// 整车跟踪：把每帧的装甲板量测送进 EKF，并处理匹配。
//
// 本作业只跟踪一辆车（config 里的 tracker.target_id），不做目标选择。
//
// 关键设计：换板用「相位」表达，而不是硬掰状态。
//   同一辆车的四块板数字相同，任何一块都可能被看到；把 phi = j*90° 显式
//   写进观测方程 h(x, phi)，换板时只改 phi，状态 (中心/速度/朝向/半径) 连续，
//   v_yaw 不会被 ±90° 的台阶污染。

#include "config.hpp"
#include "ekf.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace rm {

// 一帧里的一块装甲板，已经转换到世界系
struct Detection {
    int number = -1;        // 类别下标（= 车的编号）
    std::string label;      // 号码文本
    double conf = 0.0;
    bool is_large = false;  // 大/小装甲板（按灯条矩形宽高比判）

    cv::Vec3d p_cam{0, 0, 0};    // 相机系位置（构造量测噪声 R 要用到 Z）
    cv::Vec3d p_world{0, 0, 0};  // 世界系位置
    double yaw_world = 0.0;      // 世界系朝向（指向"板→中心"方向）
    double yaw_cam = 0.0;        // 相机系下的同一朝向（调试用）
    // 法线与光轴的夹角（弧度）。0 = 正对，越大越"相切"。
    // 板越接近相切，投影宽度越小、PnP 的深度误差按 1/cos^2 放大，
    // 位置会被放大成几十厘米的抖动 —— 所以选板要选这个角度最小的。
    double obliquity = 0.0;
    cv::Vec3d n_cam{0,0,0}, n_world{0,0,0};  // 法线（调试用）

    double range = 0.0;       // 到相机的距离
    double reproj_err = -1.0; // 重投影误差（像素）
};

// 一帧的跟踪结果
struct TrackResult {
    bool tracking = false;  // 是否已有状态
    bool detected = false;  // 本帧是否看到了主目标
    bool updated = false;   // 本帧是否真的做了卡尔曼更新
    bool switched = false;  // 本帧观测到的板号与上一帧不同（换板）
    int phase_idx = 0;      // 本帧观测到的是第几块板
    double nis = 0.0;       // 本帧新息（被拒绝时也记录，便于事后统计）

    cv::Mat x;  // 9x1
    cv::Mat P;  // 9x9

    bool is_large = false;  // 当前观测的板是大板还是小板
};

// 把角度折回 (-pi, pi]
double wrapAngle(double a);

class Tracker {
public:
    explicit Tracker(const Config& cfg);

    // dt 秒一步。K 为去畸变后的内参，R_wc 为相机→世界旋转（用于构造量测噪声）
    TrackResult step(double dt, const std::vector<Detection>& dets,
                     const cv::Matx33d& R_wc, const cv::Mat& K);

    bool initialized() const { return initialized_; }
    int targetNumber() const { return target_num_; }

private:
    // 按角点噪声 + 姿态误差 + 未建模项推量测协方差。
    cv::Mat buildMeasNoise(const Detection& d, const cv::Matx33d& R_wc,
                           const cv::Mat& K, cv::Mat* R_pos_out) const;

    Config cfg_;
    Ekf ekf_;
    int target_num_ = -1;

    bool initialized_ = false;
    bool is_large_ = false;
    int anchor_miss_ = 0;   // 当前锚定板连续丢失的帧数
    double phase_ = 0.0;    // 锚定板相对状态 yaw 的相位（0/90/180/270 度）
    double lost_dt_ = 0.0;
};

}  // namespace rm
