// 主程序：检测 → PnP → 世界系 → EKF 整车状态估计 → 输出。
//
// 流程：
//   1) 去畸变、检测网络给出四个角点、solvePnP 得到相机系位姿
//   2) 用同一帧时刻的 IMU 姿态把位姿旋到世界系（含 9 ms 接收延时补偿）
//   3) 九维 EKF：预测 → 匹配 → 更新 / 跳变 / 不更新
//   4) 输出：带状态面板与四板重投影的视频、九状态曲线图、状态 CSV

#include "camera.hpp"
#include "config.hpp"
#include "ekf.hpp"
#include "imu_reader.hpp"
#include "pnp_solver.hpp"
#include "pose_detector.hpp"
#include "reprojector.hpp"
#include "state_log.hpp"
#include "tracker.hpp"
#include "visualizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace rm;

namespace {

void printUsage() {
    std::cout
        << "用法:\n"
        << "  auto_aim --video                  处理配置中的视频（唯一模式）\n"
        << "  可选: --in <路径>  --out <路径>  --config <路径>\n";
}

// 单帧检测 + 解算（相机系）
std::vector<Armor> processFrame(const cv::Mat& undist, const cv::Mat& K,
                                PoseDetector& detector, const PnPSolver& solver,
                                const Config& cfg) {
    std::vector<Armor> out;
    for (auto& a : detector.detect(undist)) {
        if (a.number >= 0 && a.number < static_cast<int>(cfg.labels.size())) {
            a.number_label = cfg.labels[static_cast<size_t>(a.number)];
        }
        if (a.digit_conf < static_cast<float>(cfg.thresh.min_class_conf)) {
            a.number_label = "?";
        }
        if (!solver.solve(a, K)) continue;
        out.push_back(a);
    }
    return out;
}

// 相机系位姿 → 世界系量测
Detection toDetection(const Armor& a, const cv::Matx33d& R_wc) {
    Detection d;
    d.number = a.number;
    d.label = a.number_label;
    d.conf = a.digit_conf;
    d.is_large = a.is_large;
    d.reproj_err = a.reproj_err;

    d.p_cam = a.tvec;
    d.p_world = R_wc * d.p_cam;
    d.range = cv::norm(d.p_cam);

    // 板的朝向：取板坐标系的 z 轴（法线）。
    // 我们只能看到板的正面，所以法线必须指向相机（相机系下 z < 0），
    // 用这一条把 ±180° 的二义性定死。
    cv::Matx33d R_ca;
    cv::Rodrigues(a.rvec, R_ca);
    cv::Vec3d n_cam = R_ca * cv::Vec3d(0.0, 0.0, 1.0);
    if (n_cam[2] > 0.0) n_cam = -n_cam;

    const cv::Vec3d n_world = R_wc * n_cam;
    // 状态里的 yaw 指向"板 → 车中心"，与 h(x) 的减号配套，所以取反
    d.yaw_world = std::atan2(-n_world[1], -n_world[0]);
    d.yaw_cam = std::atan2(-n_cam[1], -n_cam[0]);

    // 正对度：法线与光轴（相机 z 轴）的夹角。0 = 正对，越大越接近相切。
    // 相机系下法线约等于 (0,0,-1) 表示正对，所以直接看 |z| 分量。
    const double cos_ob = std::min(1.0, std::fabs(n_cam[2]) / std::max(1e-6, cv::norm(n_cam)));
    d.obliquity = std::acos(cos_ob);
    d.n_cam = n_cam;
    d.n_world = n_world;

    return d;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string in, out, cfg_path;

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&](const char* name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("参数缺少取值: ") + name);
                }
                return argv[++i];
            };

            if (a == "--video")            continue;
            else if (a == "--in")          in = next("--in");
            else if (a == "--out")         out = next("--out");
            else if (a == "--config")      cfg_path = next("--config");
            else if (a == "-h" || a == "--help") { printUsage(); return 0; }
            else {
                std::cerr << "未知参数: " << a << "\n";
                printUsage();
                return 2;
            }
        }

        if (cfg_path.empty()) {
            cfg_path = std::string(RM_PROJECT_ROOT) + "/config/config_video.yaml";
        }

        Config cfg = Config::load(cfg_path);
        cfg.loadLabels();

        if (cfg.imu.path.empty()) throw std::runtime_error("配置缺少 imu.path");

        ImuReader imu;
        std::string imu_err;
        if (!imu.load(cfg.imu.path, &imu_err)) throw std::runtime_error(imu_err);

        PoseDetector detector(cfg);
        if (!detector.ready()) {
            std::cerr << "[error] 检测网络不可用（请检查 " << cfg.detect.model << "）\n";
            return 1;
        }
        PnPSolver solver(cfg);
        Tracker tracker(cfg);

        const std::string video_in = in.empty() ? cfg.video_in : in;
        const std::string video_out = out.empty() ? cfg.video_out : out;

        cv::VideoCapture cap(video_in);
        if (!cap.isOpened()) {
            std::cerr << "[error] 打不开视频: " << video_in << "\n";
            return 1;
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[error] 视频没有可读帧\n";
            return 1;
        }

        const Undistorter undist(cfg.camera, frame.size());
        if (!undist.valid()) {
            std::cerr << "[error] 相机内参无效\n";
            return 1;
        }
        const cv::Mat K = undist.K();

        std::cout << "[info] 图像 " << frame.cols << "x" << frame.rows
                  << " | 去畸变内参 fx=" << K.at<double>(0, 0)
                  << " fy=" << K.at<double>(1, 1) << "\n"
                  << "[info] IMU 行数 " << imu.size()
                  << " | 主目标 = " << cfg.tracker.target_id
                  << " (类别 " << tracker.targetNumber() << ")\n";

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 1.0) fps = 30.0;

        cv::VideoWriter writer;
        if (!video_out.empty()) {
            const fs::path p(video_out);
            if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
            writer.open(video_out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
            if (!writer.isOpened()) {
                std::cerr << "[warn] mp4v 不可用，改用 avc1\n";
                writer.open(video_out, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame.size());
            }
            if (!writer.isOpened()) {
                std::cerr << "[warn] 视频写入器打开失败，只处理不保存\n";
            } else {
                std::cout << "[info] 输出视频: " << video_out << "\n";
            }
        }

        // 世界 ← 机体 用 IMU 姿态；机体 ← 相机 用固定外参
        // （外参未给，按"相机光轴与机体前向一致"处理）
        const cv::Matx33d R_bc = ImuReader::cameraToBody();

        std::vector<FrameRecord> records;
        records.reserve(600);

        int idx = 0;
        int updates = 0, switches = 0, track_frames = 0;
        double nis_sum = 0.0;
        int nis_cnt = 0, nis_reject = 0;

        const auto t0 = std::chrono::steady_clock::now();

        while (!frame.empty()) {
            cv::Mat img = undist.apply(frame);

            // 1) 检测 + PnP（相机系）
            const auto armors = processFrame(img, K, detector, solver, cfg);

            // 2) 该帧时刻的姿态 → 世界系
            const Quat q = imu.attitudeForFrame(static_cast<size_t>(idx), cfg.imu.recv_latency);
            const cv::Matx33d R_wb = ImuReader::rotationFromQuat(q);
            const cv::Matx33d R_wc = R_wb * R_bc;
            const cv::Matx33d R_cw = R_wc.t();

            std::vector<Detection> dets;
            dets.reserve(armors.size());
            for (const auto& a : armors) dets.push_back(toDetection(a, R_wc));

            // 3) EKF 一步
            const double dt = imu.dtOfFrame(static_cast<size_t>(idx));
            const TrackResult res = tracker.step(dt, dets, R_wc, K);

            if (res.updated) ++updates;
            if (res.switched) ++switches;
            if (res.tracking) ++track_frames;
            if (res.detected) {
                nis_sum += res.nis;
                ++nis_cnt;
                if (!res.updated) ++nis_reject;
            }

            // 4) 绘制
            for (const auto& a : armors) Visualizer::drawArmor(img, a, K);
            if (res.tracking) {
                Reprojector::drawEstimatedArmors(img, K, R_cw, res.x,
                                                 res.is_large, res.phase_idx, cfg);
            }
            Reprojector::drawStatePanel(img, res.x, res.P, idx, res.tracking,
                                        res.switched, res.nis);
            Visualizer::drawHud(img, idx, static_cast<int>(armors.size()));

            if (writer.isOpened()) writer.write(img);

            // 5) 记录
            FrameRecord rec;
            rec.frame = idx;
            rec.t = static_cast<double>(idx) * (imu.size() > 1 ? imu.dtOfFrame(1) : 0.033);
            rec.tracking = res.tracking ? 1 : 0;
            rec.updated = res.updated ? 1 : 0;
            rec.switched = res.switched ? 1 : 0;
            rec.nis = res.nis;
            if (res.tracking && !res.x.empty()) {
                for (int i = 0; i < 9; ++i) {
                    rec.x[i] = res.x.at<double>(i);
                    rec.var[i] = std::max(0.0, res.P.at<double>(i, i));
                }
            }
            records.push_back(rec);

            ++idx;
            if (!cap.read(frame)) break;
        }

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();

        std::cout << "[done] 共 " << idx << " 帧, 平均 "
                  << (idx > 0 ? static_cast<double>(ms) / idx : 0.0) << " ms/帧\n"
                  << "[stat] 有状态 " << track_frames << " 帧 | 更新 " << updates
                  << " 次 | 换板 " << switches << " 次\n";

        if (nis_cnt > 0) {
            std::cout << "[stat] NIS 均值 " << (nis_sum / nis_cnt)
                      << " | 超门控 " << nis_reject << "/" << nis_cnt
                      << " (" << (100.0 * nis_reject / nis_cnt) << "%)\n";
        }

        if (!cfg.states_csv.empty()) {
            if (writeStatesCsv(cfg.states_csv, records))
                std::cout << "[info] 状态表: " << cfg.states_csv << "\n";
            else
                std::cerr << "[warn] 状态表写入失败: " << cfg.states_csv << "\n";
        }

        if (!cfg.states_png.empty()) {
            if (drawStateCurves(cfg.states_png, records))
                std::cout << "[info] 状态曲线: " << cfg.states_png << "\n";
            else
                std::cerr << "[warn] 状态曲线绘制失败: " << cfg.states_png << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
