#pragma once

// 状态记录与可视化：每帧的 9 个状态量、方差、NIS 存成 CSV，
// 并把 9 条曲线画进一张 PNG（带 ±3σ 带）。
//
// 曲线是"协方差是否准确"最直观的证据：估计值应落在带内，带宽也要合理。

#include <array>
#include <string>
#include <vector>

namespace rm {

struct FrameRecord {
    int frame = 0;
    double t = 0.0;
    std::array<double, 9> x{};
    std::array<double, 9> var{};
    double nis = 0.0;
    int tracking = 0;
    int updated = 0;
    int switched = 0;  // 本帧观测到的板号与上一帧不同
};

bool writeStatesCsv(const std::string& path, const std::vector<FrameRecord>& recs);

bool drawStateCurves(const std::string& path, const std::vector<FrameRecord>& recs);

}  // namespace rm
