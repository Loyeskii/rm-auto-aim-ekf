#include "config.hpp"

#include <opencv2/core.hpp>
#include <fstream>
#include <stdexcept>

namespace rm {

namespace {

// 把配置里的相对路径转成绝对路径（相对于工程根目录）
std::string resolvePath(const std::string& p) {
    if (p.empty()) return p;
    if (p.size() > 1 && (p[1] == ':' || p[0] == '/' || p[0] == '\\')) return p;
    return std::string(RM_PROJECT_ROOT) + "/" + p;
}

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

}  // namespace

Config Config::load(const std::string& yaml_path) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("打不开配置文件: " + yaml_path);
    }

    Config c;

    // ---- 相机内参 ----
    // 说明：OpenCV FileStorage 把 YAML 数组读进 Mat 的行为不稳定，
    //       所以配置里把内参拆成独立标量，这里再拼成矩阵。
    const cv::FileNode cam = fs["camera"];
    if (cam.empty()) throw std::runtime_error("配置缺少 camera 段");
    c.camera.calib_width = static_cast<int>(cam["calib_width"]);
    c.camera.calib_height = static_cast<int>(cam["calib_height"]);
    const double fx = static_cast<double>(cam["fx"]);
    const double fy = static_cast<double>(cam["fy"]);
    const double cx = static_cast<double>(cam["cx"]);
    const double cy = static_cast<double>(cam["cy"]);
    c.camera.K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                           0.0, fy, cy,
                                           0.0, 0.0, 1.0);
    c.camera.D = (cv::Mat_<double>(1, 5) << static_cast<double>(cam["k1"]),
                                             static_cast<double>(cam["k2"]),
                                             static_cast<double>(cam["p1"]),
                                             static_cast<double>(cam["p2"]),
                                             static_cast<double>(cam["k3"]));

    // ---- 装甲板尺寸 ----
    const cv::FileNode as = fs["armor_size"];
    if (!as.empty()) {
        c.armor_size.small_w = static_cast<double>(as["small_w"]);
        c.armor_size.small_h = static_cast<double>(as["small_h"]);
        c.armor_size.large_w = static_cast<double>(as["large_w"]);
        c.armor_size.large_h = static_cast<double>(as["large_h"]);
    }

    // ---- 检测网络 ----
    const cv::FileNode dt = fs["detect"];
    if (!dt.empty()) {
        c.detect.model = resolvePath(static_cast<std::string>(dt["model"]));
        if (!dt["input_w"].empty())    c.detect.input_w = static_cast<int>(dt["input_w"]);
        if (!dt["input_h"].empty())    c.detect.input_h = static_cast<int>(dt["input_h"]);
        if (!dt["conf_thres"].empty()) c.detect.conf_thres = static_cast<double>(dt["conf_thres"]);
        if (!dt["nms_thres"].empty())  c.detect.nms_thres = static_cast<double>(dt["nms_thres"]);
        if (!dt["max_dets"].empty())   c.detect.max_dets = static_cast<int>(dt["max_dets"]);
        if (!dt["swap_rb"].empty())    c.detect.swap_rb = (static_cast<int>(dt["swap_rb"]) != 0);
        if (!dt["large_aspect"].empty())
            c.detect.large_aspect = static_cast<double>(dt["large_aspect"]);
    }

    // ---- 阈值 ----
    const cv::FileNode th = fs["thresh"];
    if (!th.empty()) {
        if (!th["max_range_m"].empty())
            c.thresh.max_range_m = static_cast<double>(th["max_range_m"]);
        if (!th["min_class_conf"].empty())
            c.thresh.min_class_conf = static_cast<double>(th["min_class_conf"]);
    }

    // ---- 跟踪 / 匹配 ----
    const cv::FileNode tk = fs["tracker"];
    if (!tk.empty()) {
        if (!tk["target_id"].empty())
            c.tracker.target_id = static_cast<std::string>(tk["target_id"]);
        if (!tk["max_match_distance"].empty())
            c.tracker.max_match_distance = static_cast<double>(tk["max_match_distance"]);
        if (!tk["max_match_yaw_diff"].empty())
            c.tracker.max_match_yaw_diff = static_cast<double>(tk["max_match_yaw_diff"]);
        if (!tk["max_reproj_err"].empty())
            c.tracker.max_reproj_err = static_cast<double>(tk["max_reproj_err"]);
        if (!tk["initial_radius"].empty())
            c.tracker.initial_radius = static_cast<double>(tk["initial_radius"]);
        if (!tk["lost_time_thres"].empty())
            c.tracker.lost_time_thres = static_cast<double>(tk["lost_time_thres"]);
        if (!tk["radius_min"].empty())
            c.tracker.radius_min = static_cast<double>(tk["radius_min"]);
        if (!tk["radius_max"].empty())
            c.tracker.radius_max = static_cast<double>(tk["radius_max"]);
        if (!tk["armors_num"].empty())
            c.tracker.armors_num = static_cast<int>(tk["armors_num"]);
        if (!tk["switch_frames"].empty())
            c.tracker.switch_frames = static_cast<int>(tk["switch_frames"]);
        if (!tk["switch_margin_deg"].empty())
            c.tracker.switch_margin_deg = static_cast<double>(tk["switch_margin_deg"]);
        if (!tk["pitch"].empty())
            c.tracker.pitch = static_cast<double>(tk["pitch"]);
    }

    // ---- EKF ----
    const cv::FileNode ek = fs["ekf"];
    if (!ek.empty()) {
        if (!ek["sigma2_q_xyz"].empty())
            c.ekf.sigma2_q_xyz = static_cast<double>(ek["sigma2_q_xyz"]);
        if (!ek["sigma2_q_yaw"].empty())
            c.ekf.sigma2_q_yaw = static_cast<double>(ek["sigma2_q_yaw"]);
        if (!ek["sigma2_q_r"].empty())
            c.ekf.sigma2_q_r = static_cast<double>(ek["sigma2_q_r"]);
        if (!ek["sigma_px"].empty())
            c.ekf.sigma_px = static_cast<double>(ek["sigma_px"]);
        if (!ek["sigma_att_deg"].empty())
            c.ekf.sigma_att_deg = static_cast<double>(ek["sigma_att_deg"]);
        if (!ek["sigma_world_floor"].empty())
            c.ekf.sigma_world_floor = static_cast<double>(ek["sigma_world_floor"]);
        if (!ek["sigma_yaw_deg"].empty())
            c.ekf.sigma_yaw_deg = static_cast<double>(ek["sigma_yaw_deg"]);
        if (!ek["gate_pos"].empty())
            c.ekf.gate_pos = static_cast<double>(ek["gate_pos"]);
        if (!ek["nis_gate"].empty())
            c.ekf.nis_gate = static_cast<double>(ek["nis_gate"]);
        if (!ek["p0_sigma_pos"].empty())
            c.ekf.p0_sigma_pos = static_cast<double>(ek["p0_sigma_pos"]);
        if (!ek["p0_sigma_vel"].empty())
            c.ekf.p0_sigma_vel = static_cast<double>(ek["p0_sigma_vel"]);
        if (!ek["p0_sigma_za"].empty())
            c.ekf.p0_sigma_za = static_cast<double>(ek["p0_sigma_za"]);
        if (!ek["p0_sigma_vza"].empty())
            c.ekf.p0_sigma_vza = static_cast<double>(ek["p0_sigma_vza"]);
        if (!ek["p0_sigma_yaw"].empty())
            c.ekf.p0_sigma_yaw = static_cast<double>(ek["p0_sigma_yaw"]);
        if (!ek["p0_sigma_vyaw"].empty())
            c.ekf.p0_sigma_vyaw = static_cast<double>(ek["p0_sigma_vyaw"]);
        if (!ek["p0_sigma_r"].empty())
            c.ekf.p0_sigma_r = static_cast<double>(ek["p0_sigma_r"]);
    }

    // ---- IMU ----
    const cv::FileNode im = fs["imu"];
    if (!im.empty()) {
        c.imu.path = resolvePath(static_cast<std::string>(im["path"]));
        if (!im["recv_latency"].empty())
            c.imu.recv_latency = static_cast<double>(im["recv_latency"]);
    }

    // ---- 路径 ----
    const cv::FileNode md = fs["model"];
    const cv::FileNode io = fs["io"];
    c.label_file = resolvePath(static_cast<std::string>(md["labels"]));
    c.video_in   = resolvePath(static_cast<std::string>(io["video"]));
    c.video_out  = resolvePath(static_cast<std::string>(io["video_out"]));
    if (!io["states_csv"].empty())
        c.states_csv = resolvePath(static_cast<std::string>(io["states_csv"]));
    if (!io["states_png"].empty())
        c.states_png = resolvePath(static_cast<std::string>(io["states_png"]));

    return c;
}

int Config::targetNumber() const {
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == tracker.target_id) return static_cast<int>(i);
    }
    return -1;
}

void Config::loadLabels() {
    labels.clear();
    std::ifstream ifs(label_file);
    if (!ifs) {
        throw std::runtime_error("打不开标签文件: " + label_file);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        const std::string t = trim(line);
        if (!t.empty()) labels.push_back(t);
    }
    if (labels.empty()) {
        throw std::runtime_error("标签文件为空: " + label_file);
    }
}

}  // namespace rm
