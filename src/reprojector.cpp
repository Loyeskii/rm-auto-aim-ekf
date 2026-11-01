#include "reprojector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace rm {

namespace {

// 一块板在世界系下的四个角点。
// yaw_state 指向"板 -> 车中心"，所以板的外法线方向是 yaw_state + pi。
std::vector<cv::Point3d> plateCorners(const cv::Vec3d& center, double yaw_state,
                                      double pitch, double W, double H) {
    const double yout = yaw_state + M_PI;  // 板的外法线方向
    const double cy = std::cos(yout), sy = std::sin(yout);
    const double cp = std::cos(pitch), sp = std::sin(pitch);

    // 由 R = Rz(yout)*Ry(pitch) 展开得到的三个轴：
    //   R*(1,0,0) = 法线（近水平，朝车外）
    //   R*(0,1,0) = 板宽（水平切向）
    //   R*(0,0,1) = 板高（近竖直，pitch=0 时正好是 +z）
    // 板面由"板宽 × 板高"张成，法线垂直于板面。
    const cv::Vec3d a_w(-sy, cy, 0.0);          // 沿板宽
    const cv::Vec3d a_h(sp * cy, sp * sy, cp);  // 沿板高

    const double hw = W * 0.5, hh = H * 0.5;
    std::vector<cv::Point3d> c(4);
    c[0] = cv::Point3d(center + hw * a_w + hh * a_h);
    c[1] = cv::Point3d(center + hw * a_w - hh * a_h);
    c[2] = cv::Point3d(center - hw * a_w - hh * a_h);
    c[3] = cv::Point3d(center - hw * a_w + hh * a_h);
    return c;
}

bool project(const cv::Vec3d& pw, const cv::Matx33d& R_cw, const cv::Mat& K,
             cv::Point2f* out) {
    const cv::Vec3d pc = R_cw * pw;
    if (pc[2] < 0.05) return false;  // 在相机背后或过近
    const double u = K.at<double>(0, 0) * pc[0] / pc[2] + K.at<double>(0, 2);
    const double v = K.at<double>(1, 1) * pc[1] / pc[2] + K.at<double>(1, 2);
    *out = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
    return true;
}

}  // namespace

void Reprojector::drawEstimatedArmors(cv::Mat& img, const cv::Mat& K,
                                      const cv::Matx33d& R_cw, const cv::Mat& x,
                                      bool is_large, int phase_idx, const Config& cfg) {
    if (x.empty()) return;

    const double xc = x.at<double>(0);
    const double yc = x.at<double>(2);
    const double za = x.at<double>(4);
    const double yaw = x.at<double>(6);
    const double r = x.at<double>(8);

    const double W = is_large ? cfg.armor_size.large_w : cfg.armor_size.small_w;
    const double H = is_large ? cfg.armor_size.large_h : cfg.armor_size.small_h;

    const int n = std::max(2, cfg.tracker.armors_num);
    for (int j = 0; j < n; ++j) {
        const double phi = 2.0 * M_PI * j / n;
        const double yaw_j = yaw + phi;

        // 四块板都按同一半径、同一高度建模（差异小于 r 的不确定度）
        const cv::Vec3d center(xc - r * std::cos(yaw_j),
                               yc - r * std::sin(yaw_j), za);

        const auto corners = plateCorners(center, yaw_j, cfg.tracker.pitch, W, H);

        std::vector<cv::Point> poly;
        poly.reserve(4);
        bool ok = true;
        for (const auto& c : corners) {
            cv::Point2f p;
            if (!project(c, R_cw, K, &p)) { ok = false; break; }
            poly.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        }
        if (!ok) continue;

        // 当前观测到的那块板画黄色加粗，其余画绿色
        const bool cur = (j == phase_idx);
        const cv::Scalar color = cur ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
        cv::polylines(img, poly, true, color, cur ? 3 : 2, cv::LINE_AA);

        char buf[32];
        std::snprintf(buf, sizeof(buf), "P%d", j);
        cv::putText(img, buf, poly[0] + cv::Point(2, -4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv::LINE_AA);
    }
}

void Reprojector::drawStatePanel(cv::Mat& img, const cv::Mat& x, const cv::Mat& P,
                                 int frame, bool tracking, bool switched, double nis) {
    const int w = 340;
    const int line_h = 22;
    const int rows = 9;
    const int h = 34 + 22 + rows * line_h + 30;

    const int x0 = std::max(0, img.cols - w - 12);
    const int y0 = 12;

    cv::Mat roi = img(cv::Rect(x0, y0, w, h));
    cv::Mat panel(roi.size(), roi.type(), cv::Scalar(28, 28, 28));
    cv::addWeighted(panel, 0.78, roi, 0.22, 0.0, roi);
    cv::rectangle(img, cv::Rect(x0, y0, w, h), cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    char buf[128];

    if (!tracking || x.empty()) {
        cv::putText(img, "frame", cv::Point(x0 + 12, y0 + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        std::snprintf(buf, sizeof(buf), "%d   NO TARGET", frame);
        cv::putText(img, buf, cv::Point(x0 + 70, y0 + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(120, 120, 255), 1, cv::LINE_AA);
        return;
    }

    std::snprintf(buf, sizeof(buf), "frame %d%s", frame, switched ? "  SWITCH" : "");
    cv::putText(img, buf, cv::Point(x0 + 12, y0 + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    std::snprintf(buf, sizeof(buf), "NIS %.1f", nis);
    cv::putText(img, buf, cv::Point(x0 + 190, y0 + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(160, 200, 160), 1, cv::LINE_AA);

    static const char* kNames[9] = {"xc", "v_xc", "yc", "v_yc", "za",
                                    "v_za", "yaw", "v_yaw", "r"};

    for (int i = 0; i < rows; ++i) {
        const double v = x.at<double>(i);
        const double s = (P.empty() || i >= P.rows) ? 0.0 : std::sqrt(std::max(0.0, P.at<double>(i, i)));

        std::snprintf(buf, sizeof(buf), "%-6s %8.3f", kNames[i], v);
        cv::putText(img, buf, cv::Point(x0 + 12, y0 + 48 + i * line_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

        // 标准差另起一小段，颜色偏暗，和估计值区分开
        std::snprintf(buf, sizeof(buf), "+-%.3f", s);
        cv::putText(img, buf, cv::Point(x0 + 220, y0 + 48 + i * line_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(150, 170, 200), 1, cv::LINE_AA);
    }
}

}  // namespace rm
