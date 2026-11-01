#pragma once

// 配置结构与加载（神经网络检测 + 整车状态 EKF）。
//
// 设计原则与上周一致：所有阈值、路径都放在 config 里，改参数不需要重新编译。

#include "types.hpp"
#include <string>
#include <vector>

namespace rm {

struct Config {
    // ---- 相机 ----
    CameraIntrinsic camera;

    // ---- 装甲板真实尺寸（米）----
    // 填的是「灯条外角矩形」：宽 = 两根灯条外沿间距，高 = 灯条长度。
    // 不是模块外形（140×125 / 235×127），也不是整块板的高度（125/127）。
    // 网络关键点落在灯条外角上，PnP 物方点必须与之同源。
    struct ArmorSize {
        double small_w = 0.135;
        double small_h = 0.055;
        double large_w = 0.230;
        double large_h = 0.055;
    } armor_size;

    // ---- 检测网络 ----
    struct Detect {
        std::string model;          // onnx 路径
        int input_w = 640;          // 网络输入宽（与导出时的 imgsz 一致）
        int input_h = 480;
        double conf_thres = 0.25;   // 类别置信度阈值
        double nms_thres = 0.45;    // NMS 的 IoU 阈值
        int max_dets = 20;          // NMS 后最多保留多少个
        bool swap_rb = true;        // 网络用 RGB 训练 → BGR 转 RGB
        // 宽高比超过该值判为大板。灯条矩形比例：小板 2.45、大板 4.18
        double large_aspect = 2.5;
    } detect;

    // ---- 阈值 ----
    struct Thresh {
        double max_range_m = 15.0;    // 解算距离上限，超过视为无效解
        double min_class_conf = 0.30; // 低于该置信度视为"号码不可信"
    } thresh;

    // ---- 跟踪 / 匹配 ----
    struct Tracker {
        // 主目标号码。本作业直接指定，不做目标选择：
        // 实战中每辆敌车各维护一个滤波器，作业里只实例化一个。
        std::string target_id = "4";

        double max_match_distance = 0.15;  // 位置门限（米）
        double max_match_yaw_diff = 1.0;   // 朝向门限（弧度），跳变判据也用它
        double max_reproj_err = 5.0;       // 重投影误差门限（像素），粗差剔除
        double initial_radius = 0.26;      // 初始化时假设的板到中心距离（米）
        double lost_time_thres = 0.3;      // 连续多久没更新算丢失（秒）
        double radius_min = 0.12;          // 半径钳位下限，防止发散
        double radius_max = 0.40;          // 半径钳位上限
        int armors_num = 4;                // 标准步兵 4 块板
        // 换锚确认帧数：当前锚定的板连续丢失这么多帧、且另一块连续可见，
        // 才允许把锚点切过去。四块板间隔 90°，yaw 模 90° 不可观，
        // 若每帧自由选板，状态 yaw 会沿这个自由度滑走。
        int switch_frames = 5;
        // 正对度余量（度）：新板的正对度比当前锚定板好出这么多才切换，
        // 否则两块角度接近时会来回抖。
        double switch_margin_deg = 10.0;
        double pitch = 0.0;                // 重投影时板的俯仰角（弧度），真实板近似竖直
    } tracker;

    // ---- EKF ----
    struct Ekf {
        // 过程噪声（连续白噪声加速度模型），量纲见报告
        double sigma2_q_xyz = 20.0;
        double sigma2_q_yaw = 100.0;
        double sigma2_q_r = 800.0;

        // 角点噪声（像素），用来推量测噪声 R
        double sigma_px = 1.0;

        // 世界系量测的主要噪声来源并不是角点，而是：
        //   1) 姿态误差（IMU 帧间抖动），jitter 直接以 |p| 为力臂放大成位置误差
        //   2) 自车平移未建模（素材只有姿态，没有定位）
        double sigma_att_deg = 3.0;      // 姿态标准差（度）
        double sigma_world_floor = 0.08; // 世界系位置噪声下限（米）

        // 朝向量测的标准差下限（度）。
        // 四点 PnP 解出的板朝向本身就不准（薄板、角点噪声、姿态误差串联），
        // 实测 NIS 统计表明它远大于"角点噪声/像素宽度"这个理想值。
        double sigma_yaw_deg = 6.0;

        // 新息门控
        double nis_gate = 9.488;   // chi2(4, 0.95)，四维量测
        double gate_pos = 7.815;   // chi2(3, 0.95)，位置三项的马氏门

        // 初始协方差的对角标准差（不是方差）
        double p0_sigma_pos = 0.10;    // xc, yc
        double p0_sigma_vel = 1.00;    // v_xc, v_yc
        double p0_sigma_za = 0.05;     // za
        double p0_sigma_vza = 0.50;    // v_za
        double p0_sigma_yaw = 0.20;    // yaw
        double p0_sigma_vyaw = 2.00;   // v_yaw
        double p0_sigma_r = 0.05;      // r
    } ekf;

    // ---- IMU ----
    struct Imu {
        std::string path;
        // 数据里是"收到时间戳"，需要回退这个量才是采集时刻。
        // 正值表示取 (t_图像 - 值) 处的姿态。
        double recv_latency = 0.009;
    } imu;

    // ---- 输出 ----
    std::string states_csv;
    std::string states_png;

    // ---- 标签 ----
    std::string label_file;
    std::vector<std::string> labels;

    // ---- 输入输出 ----
    std::string video_in;
    std::string video_out;

    static Config load(const std::string& yaml_path);
    void loadLabels();

    // target_id 字符串 → 类别下标；找不到返回 -1
    int targetNumber() const;
};

}  // namespace rm
