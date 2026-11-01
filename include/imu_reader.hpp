#pragma once

// IMU 内录数据读取与姿态插值。
//
// 数据格式（IMU.txt）：每行 5 列，空格分隔，无表头
//   t  qw  qx  qy  qz
// 共 531 行，与视频 531 帧一一对应；采样间隔 41.5 ms（≈24 Hz）。
//
// 关键点：时间戳是**接收时间**，不是采集时间。所以第 i 帧要用的姿态，
//        是 (t_i - latency) 时刻的四元数，需要在前一行与本行之间做 slerp。
//        9 ms 只有 0.217 个采样间隔，直接取最近一行会把修正量化掉。

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace rm {

struct Quat {
    double w = 1.0, x = 0.0, y = 0.0, z = 0.0;
};

// 单位四元数球面线性插值
Quat slerp(const Quat& a, const Quat& b, double u);

class ImuReader {
public:
    bool load(const std::string& path, std::string* err = nullptr);

    size_t size() const { return t_.size(); }
    bool empty() const { return t_.empty(); }

    // 第 i 帧（对应第 i 行）在 (t_i - latency) 时刻的姿态
    Quat attitudeForFrame(size_t i, double latency) const;

    // 第 i 帧与前一帧的时间差（秒）；i=0 时退化为均匀间隔
    double dtOfFrame(size_t i) const;

    // 四元数 → 旋转矩阵（机体 → 世界）
    static cv::Matx33d rotationFromQuat(const Quat& q);

    // 相机系 → 机体系 的固定旋转。
    // 相机：x 右 / y 下 / z 前；机体：x 前 / y 左 / z 上。
    // 列向量为相机三轴在机体下的表示：x_cam=(0,-1,0) y_cam=(0,0,-1) z_cam=(1,0,0)
    static cv::Matx33d cameraToBody();

private:
    std::vector<double> t_;
    std::vector<Quat> q_;
};

}  // namespace rm
