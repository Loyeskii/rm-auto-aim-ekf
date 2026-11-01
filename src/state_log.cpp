#include "state_log.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

namespace rm {

namespace {

const char* kNames[9] = {"xc", "v_xc", "yc", "v_yc", "za", "v_za", "yaw", "v_yaw", "r"};
const char* kUnits[9] = {"m", "m/s", "m", "m/s", "m", "m/s", "rad", "rad/s", "m"};

// 九条曲线的公共绘图参数
const int kCanvasW = 1800;
const int kCanvasH = 1200;
const int kMarginL = 80;
const int kMarginT = 60;
const int kPlotW = 540;
const int kPlotH = 330;
const int kGapX = 40;
const int kGapY = 60;

}  // namespace

bool writeStatesCsv(const std::string& path, const std::vector<FrameRecord>& recs) {
    std::ofstream ofs(path);
    if (!ofs) return false;

    ofs << "frame,t";
    for (int i = 0; i < 9; ++i) ofs << ",x_" << kNames[i];
    for (int i = 0; i < 9; ++i) ofs << ",var_" << kNames[i];
    ofs << ",nis,tracking,updated,switched\n";

    char buf[512];
    for (const auto& r : recs) {
        std::snprintf(buf, sizeof(buf),
                      "%d,%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
                      "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                      "%.3f,%d,%d,%d\n",
                      r.frame, r.t,
                      r.x[0], r.x[1], r.x[2], r.x[3], r.x[4], r.x[5], r.x[6], r.x[7], r.x[8],
                      r.var[0], r.var[1], r.var[2], r.var[3], r.var[4],
                      r.var[5], r.var[6], r.var[7], r.var[8],
                      r.nis, r.tracking, r.updated, r.switched);
        ofs << buf;
    }
    return true;
}

bool drawStateCurves(const std::string& path, const std::vector<FrameRecord>& recs) {
    if (recs.empty()) return false;

    cv::Mat canvas(kCanvasH, kCanvasW, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::putText(canvas, "EKF states  (red = estimate, band = +-3 sigma)",
                cv::Point(kMarginL, 34), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(40, 40, 40), 2, cv::LINE_AA);

    const double f0 = recs.front().frame;
    const double f1 = std::max(f0 + 1.0, static_cast<double>(recs.back().frame));

    for (int i = 0; i < 9; ++i) {
        const int row = i / 3, col = i % 3;
        const int px = kMarginL + col * (kPlotW + kGapX);
        const int py = kMarginT + row * (kPlotH + kGapY);

        const cv::Rect box(px, py, kPlotW, kPlotH);
        cv::rectangle(canvas, box, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

        // yaw 是「累计转角」：目标持续自旋时它会一路增长（本段约 -78 rad）。
        // 直接画成一条长斜线，读者容易误以为估计发散。这一格折回 (-pi, pi]
        // 显示：斜率仍然等于 v_yaw，绝对值被约束在可读范围内。
        // 其余 8 格完全按原样绘制。
        const bool wrap_pi = (i == 6);
        auto wrapVal = [](double v) {
            double w = std::fmod(v + M_PI, 2.0 * M_PI);
            if (w < 0.0) w += 2.0 * M_PI;
            return w - M_PI;
        };
        // 图上显示的值：wrap 图先折回中心，再叠加偏移（±3σ）并截断在边界内，
        // 避免带宽跨过 ±pi 后被翻到另一侧。
        auto shown = [&](size_t k, double offset) {
            if (!wrap_pi) return recs[k].x[i] + offset;
            const double c = wrapVal(recs[k].x[i]);
            return std::max(-M_PI, std::min(M_PI, c + offset));
        };

        // 标题
        char title[96];
        if (wrap_pi)
            std::snprintf(title, sizeof(title), "%s  [%s, wrapped to +-pi]",
                          kNames[i], kUnits[i]);
        else
            std::snprintf(title, sizeof(title), "%s  [%s]", kNames[i], kUnits[i]);
        cv::putText(canvas, title, cv::Point(px, py - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);

        // 值域：把 ±3σ 也包进来，否则带宽可能被裁掉
        double vmin = std::numeric_limits<double>::max();
        double vmax = -std::numeric_limits<double>::max();
        bool any = false;
        if (wrap_pi) {
            for (const auto& r : recs) {
                if (r.tracking) { any = true; break; }
            }
            vmin = -M_PI * 1.06;   // 固定为 ±pi，读者一眼能看出「在区间内往返」
            vmax = M_PI * 1.06;
        } else {
            for (const auto& r : recs) {
                if (!r.tracking) continue;
                const double s = 3.0 * std::sqrt(std::max(0.0, r.var[i]));
                vmin = std::min(vmin, r.x[i] - s);
                vmax = std::max(vmax, r.x[i] + s);
                any = true;
            }
        }
        if (!any) {
            cv::putText(canvas, "(no data)", cv::Point(px + 20, py + kPlotH / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
            continue;
        }
        double span = vmax - vmin;
        if (span < 1e-6) span = 1.0;
        if (!wrap_pi) {
            vmin -= span * 0.08;
            vmax += span * 0.08;
            span = vmax - vmin;
        }

        auto mapX = [&](int f) {
            return px + static_cast<int>((f - f0) / (f1 - f0) * (kPlotW - 1));
        };
        auto mapY = [&](double v) {
            return py + kPlotH - 1 - static_cast<int>((v - vmin) / span * (kPlotH - 1));
        };

        // ---- ±3σ 带：按连续有效段分别填充 ----
        size_t a = 0;
        while (a < recs.size()) {
            if (!recs[a].tracking) { ++a; continue; }
            size_t b = a;
            while (b + 1 < recs.size() && recs[b + 1].tracking) {
                // wrap 图在跨越 ±pi 处断开，否则会连出一条横穿整格的竖线
                if (wrap_pi && std::fabs(wrapVal(recs[b + 1].x[i]) - wrapVal(recs[b].x[i])) > M_PI)
                    break;
                ++b;
            }

            std::vector<cv::Point> poly;
            poly.reserve((b - a + 1) * 2);
            for (size_t k = a; k <= b; ++k) {
                const double s = 3.0 * std::sqrt(std::max(0.0, recs[k].var[i]));
                poly.push_back(cv::Point(mapX(recs[k].frame), mapY(shown(k, s))));
            }
            for (size_t k = b + 1; k-- > a;) {
                const double s = 3.0 * std::sqrt(std::max(0.0, recs[k].var[i]));
                poly.push_back(cv::Point(mapX(recs[k].frame), mapY(shown(k, -s))));
            }
            if (poly.size() >= 3) {
                std::vector<std::vector<cv::Point>> polys{poly};
                cv::fillPoly(canvas, polys, cv::Scalar(215, 225, 245), cv::LINE_AA);
            }
            a = b + 1;
        }

        // ---- 估计值折线 ----
        a = 0;
        while (a < recs.size()) {
            if (!recs[a].tracking) { ++a; continue; }
            size_t b = a;
            while (b + 1 < recs.size() && recs[b + 1].tracking) {
                if (wrap_pi && std::fabs(wrapVal(recs[b + 1].x[i]) - wrapVal(recs[b].x[i])) > M_PI)
                    break;
                ++b;
            }

            std::vector<cv::Point> line;
            line.reserve(b - a + 1);
            for (size_t k = a; k <= b; ++k)
                line.push_back(cv::Point(mapX(recs[k].frame), mapY(shown(k, 0.0))));

            if (line.size() >= 2) {
                cv::polylines(canvas, line, false, cv::Scalar(0, 0, 220), 2, cv::LINE_AA);
            }
            a = b + 1;
        }

        // ---- 刻度标注 ----
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3f", vmax);
        cv::putText(canvas, buf, cv::Point(px + 4, py + 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
        std::snprintf(buf, sizeof(buf), "%.3f", vmin);
        cv::putText(canvas, buf, cv::Point(px + 4, py + kPlotH - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
        std::snprintf(buf, sizeof(buf), "frame %d-%d", recs.front().frame, recs.back().frame);
        cv::putText(canvas, buf, cv::Point(px + kPlotW - 130, py + kPlotH - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    }

    return cv::imwrite(path, canvas);
}

}  // namespace rm
