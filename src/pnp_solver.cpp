#include "pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <cmath>
#include <vector>

namespace rm {

bool PnPSolver::solve(Armor& armor, const cv::Mat& K) const {
    // ---- 1) 物方点：装甲板自身坐标系 ----
    // 原点在板中心，X 向右（宽度方向），Y 向下（高度方向），Z 为板法线。
    // 单位用米，所以 tvec 直接就是米，正好和 图N.md 的真值单位一致。
    // 顺序必须与图像点严格一致，否则解出来的位姿是错的。
    const double w = armor.is_large ? cfg_.armor_size.large_w : cfg_.armor_size.small_w;
    const double h = armor.is_large ? cfg_.armor_size.large_h : cfg_.armor_size.small_h;

    const std::vector<cv::Point3f> obj = {
        {static_cast<float>(-w / 2), static_cast<float>(-h / 2), 0.f},  // 左上
        {static_cast<float>( w / 2), static_cast<float>(-h / 2), 0.f},  // 右上
        {static_cast<float>( w / 2), static_cast<float>( h / 2), 0.f},  // 右下
        {static_cast<float>(-w / 2), static_cast<float>( h / 2), 0.f},  // 左下
    };
    const std::vector<cv::Point2f> img = {
        armor.corners[0], armor.corners[1], armor.corners[2], armor.corners[3],
    };

    // ---- 2) 求解 ----
    // 四个点共面 → 用 SOLVEPNP_IPPE（专为平面目标设计，无需初值，稳定）
    cv::Mat rvec, tvec;
    const bool ok = cv::solvePnP(obj, img, K, cv::Mat(), rvec, tvec,
                                 /*useExtrinsicGuess=*/false, cv::SOLVEPNP_IPPE);
    if (!ok) return false;

    // 去畸变后图像不再有畸变，所以 distCoeffs 传空
    const double z = tvec.at<double>(2);
    if (!std::isfinite(z) || z <= 0.05 || z > cfg_.thresh.max_range_m) {
        return false;   // 解算到相机背后或超出量程，判为无效
    }

    armor.rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    armor.tvec = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // ---- 3) 重投影误差：解算质量的体检指标 ----
    // 把物方点按解出的位姿重新投回图像，与检测到的角点比较；
    // 平均偏差越小说明解算越可信（通常应 < 2 像素）。
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, K, cv::Mat(), proj);
    double sum = 0.0;
    for (size_t i = 0; i < proj.size(); ++i) {
        sum += cv::norm(proj[i] - img[i]);
    }
    armor.reproj_err = sum / static_cast<double>(proj.size());

    armor.solved = true;
    return true;
}

}  // namespace rm
