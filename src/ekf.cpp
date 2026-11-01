#include "ekf.hpp"

#include <cmath>

namespace rm {

namespace {

cv::Mat eyeN(int n) { return cv::Mat::eye(n, n, CV_64F); }

void symmetrize(cv::Mat& P) {
    cv::Mat Pt;
    cv::transpose(P, Pt);
    P = (P + Pt) * 0.5;
}

}  // namespace

Ekf::Ekf(const Config& cfg) : cfg_(cfg) {
    x_ = cv::Mat::zeros(kStateDim, 1, CV_64F);
    P_ = cv::Mat::eye(kStateDim, kStateDim, CV_64F);
}

void Ekf::init(const cv::Vec3d& armor_pos, double armor_yaw) {
    const double r = cfg_.tracker.initial_radius;

    x_.at<double>(0) = armor_pos[0] + r * std::cos(armor_yaw);  // xc = xa + r cos
    x_.at<double>(1) = 0.0;
    x_.at<double>(2) = armor_pos[1] + r * std::sin(armor_yaw);  // yc = ya + r sin
    x_.at<double>(3) = 0.0;
    x_.at<double>(4) = armor_pos[2];
    x_.at<double>(5) = 0.0;
    x_.at<double>(6) = armor_yaw;
    x_.at<double>(7) = 0.0;
    x_.at<double>(8) = r;

    const double s[kStateDim] = {
        cfg_.ekf.p0_sigma_pos,  cfg_.ekf.p0_sigma_vel,
        cfg_.ekf.p0_sigma_pos,  cfg_.ekf.p0_sigma_vel,
        cfg_.ekf.p0_sigma_za,   cfg_.ekf.p0_sigma_vza,
        cfg_.ekf.p0_sigma_yaw,  cfg_.ekf.p0_sigma_vyaw,
        cfg_.ekf.p0_sigma_r};

    P_ = cv::Mat::zeros(kStateDim, kStateDim, CV_64F);
    for (int i = 0; i < kStateDim; ++i) P_.at<double>(i, i) = s[i] * s[i];
}

cv::Mat Ekf::processNoise(double dt) const {
    // 连续白噪声加速度模型离散化：
    //   [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * sigma^2
    const double t2 = dt * dt, t3 = t2 * dt, t4 = t3 * dt;

    const double sxyz = cfg_.ekf.sigma2_q_xyz;
    const double syaw = cfg_.ekf.sigma2_q_yaw;
    const double sr   = cfg_.ekf.sigma2_q_r;

    cv::Mat Q = cv::Mat::zeros(kStateDim, kStateDim, CV_64F);

    // 三个平移块（xc/v_xc、yc/v_yc、za/v_za）形式相同
    const int pos_idx[3] = {0, 2, 4};
    for (int b = 0; b < 3; ++b) {
        const int i = pos_idx[b];
        Q.at<double>(i,     i)     = t4 / 4.0 * sxyz;
        Q.at<double>(i,     i + 1) = t3 / 2.0 * sxyz;
        Q.at<double>(i + 1, i)     = t3 / 2.0 * sxyz;
        Q.at<double>(i + 1, i + 1) = t2 * sxyz;
    }

    // 朝向块：交叉项用 yaw 自己的 sigma（原参考实现这里用了平移的，
    // 会把 yaw-ω 相关系数从 0.5 压到 0.2，是个笔误，这里修正）
    Q.at<double>(6, 6) = t4 / 4.0 * syaw;
    Q.at<double>(6, 7) = t3 / 2.0 * syaw;
    Q.at<double>(7, 6) = t3 / 2.0 * syaw;
    Q.at<double>(7, 7) = t2 * syaw;

    // 半径：只给缓慢漂移
    Q.at<double>(8, 8) = t4 / 4.0 * sr;

    return Q;
}

void Ekf::predict(double dt) {
    // 匀速 + 匀角速：x' = F x，F 只有四个 dt 偏离单位阵
    cv::Mat F = eyeN(kStateDim);
    F.at<double>(0, 1) = dt;
    F.at<double>(2, 3) = dt;
    F.at<double>(4, 5) = dt;
    F.at<double>(6, 7) = dt;

    x_ = F * x_;
    P_ = F * P_ * F.t() + processNoise(dt);
    symmetrize(P_);
}

cv::Vec3d Ekf::predictedArmorPos(double phase) const {
    const double xc = x_.at<double>(0), yc = x_.at<double>(2);
    const double za = x_.at<double>(4);
    const double a = x_.at<double>(6) + phase;  // 第 j 块板的朝向
    const double r = x_.at<double>(8);
    return cv::Vec3d(xc - r * std::cos(a), yc - r * std::sin(a), za);
}

double Ekf::mahalanobisPos(const cv::Vec3d& p_meas, const cv::Mat& R_pos,
                           double phase) const {
    const cv::Mat Hpos = jacobianH(phase).rowRange(0, 3);  // 3x9

    const cv::Vec3d pred = predictedArmorPos(phase);
    cv::Mat innov(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) innov.at<double>(i) = p_meas[i] - pred[i];

    const cv::Mat S = Hpos * P_ * Hpos.t() + R_pos;  // 3x3
    const cv::Mat m = innov.t() * S.inv(cv::DECOMP_LU) * innov;
    return m.at<double>(0);
}

cv::Mat Ekf::jacobianH(double phase) const {
    const double a = x_.at<double>(6) + phase;  // 该板的朝向
    const double r = x_.at<double>(8);

    cv::Mat H = cv::Mat::zeros(kMeasDim, kStateDim, CV_64F);
    // x_a 行
    H.at<double>(0, 0) = 1.0;
    H.at<double>(0, 6) = r * std::sin(a);
    H.at<double>(0, 8) = -std::cos(a);
    // y_a 行
    H.at<double>(1, 2) = 1.0;
    H.at<double>(1, 6) = -r * std::cos(a);
    H.at<double>(1, 8) = -std::sin(a);
    // z_a 行
    H.at<double>(2, 4) = 1.0;
    // yaw_a 行
    H.at<double>(3, 6) = 1.0;
    return H;
}

bool Ekf::update(const cv::Vec4d& z, const cv::Mat& R, double nis_gate, double* nis,
                 double phase) {
    const cv::Mat H = jacobianH(phase);
    const double a = x_.at<double>(6) + phase;
    const double r = x_.at<double>(8);

    cv::Vec4d hx;
    hx[0] = x_.at<double>(0) - r * std::cos(a);
    hx[1] = x_.at<double>(2) - r * std::sin(a);
    hx[2] = x_.at<double>(4);
    hx[3] = a;

    cv::Mat innov(kMeasDim, 1, CV_64F);
    for (int i = 0; i < kMeasDim; ++i) innov.at<double>(i) = z[i] - hx[i];

    const cv::Mat S = H * P_ * H.t() + R;
    const cv::Mat S_inv = S.inv(cv::DECOMP_LU);

    // 新息的门控：νᵀS⁻¹ν 服从 chi2(4)
    const cv::Mat nis_mat = innov.t() * S_inv * innov;
    const double nis_val = nis_mat.at<double>(0);
    if (nis) *nis = nis_val;
    if (nis_val > nis_gate) return false;

    const cv::Mat K = P_ * H.t() * S_inv;
    x_ = x_ + K * innov;
    P_ = (cv::Mat::eye(kStateDim, kStateDim, CV_64F) - K * H) * P_;
    symmetrize(P_);
    return true;
}

void Ekf::inflateVariance(int idx, double factor) {
    // 整行整列同时乘 sqrt(f)：对角元被乘两次恰好得到 f，
    // 交叉项也被放宽，且保持对称与非负定性。
    const double s = std::sqrt(factor);
    for (int j = 0; j < kStateDim; ++j) {
        P_.at<double>(idx, j) *= s;
        P_.at<double>(j, idx) *= s;
    }
}

void Ekf::resetPosVel(const cv::Vec3d& armor_pos) {    const double yaw = x_.at<double>(6);
    const double r = x_.at<double>(8);
    x_.at<double>(0) = armor_pos[0] + r * std::cos(yaw);
    x_.at<double>(1) = 0.0;
    x_.at<double>(2) = armor_pos[1] + r * std::sin(yaw);
    x_.at<double>(3) = 0.0;
    x_.at<double>(4) = armor_pos[2];
    x_.at<double>(5) = 0.0;
}

cv::Matx33d Ekf::rotationFromYaw(double yaw) const {
    const double c = std::cos(yaw), s = std::sin(yaw);
    return cv::Matx33d(c, -s, 0, s, c, 0, 0, 0, 1);
}

}  // namespace rm
