#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace rm {

double wrapAngle(double a) {
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

Tracker::Tracker(const Config& cfg) : cfg_(cfg), ekf_(cfg) {
    target_num_ = cfg_.targetNumber();
    if (target_num_ < 0) {
        throw std::runtime_error("tracker.target_id = \"" + cfg_.tracker.target_id +
                                 "\" 在 labels.txt 里找不到对应类别");
    }
}

cv::Mat Tracker::buildMeasNoise(const Detection& d, const cv::Matx33d& R_wc,
                                const cv::Mat& K, cv::Mat* R_pos_out) const {
    // ---- ① PnP 自身的各向异性 ----
    //   横向  sigma = Z / f  * sigma_px
    //   深度  sigma = Z^2/(f*W) * sigma_px
    // 小平面目标的深度误差比横向大一个量级。
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double W = d.is_large ? cfg_.armor_size.large_w : cfg_.armor_size.small_w;
    const double spx = cfg_.ekf.sigma_px;

    const double Z = std::max(0.15, d.p_cam[2]);
    const double sx = Z / fx * spx;
    const double sy = Z / fy * spx;
    const double sz = Z * Z / (fx * W) * spx;

    cv::Mat Rc = cv::Mat::zeros(3, 3, CV_64F);
    Rc.at<double>(0, 0) = sx * sx;
    Rc.at<double>(1, 1) = sy * sy;
    Rc.at<double>(2, 2) = sz * sz;

    const cv::Mat Rm(R_wc);
    cv::Mat Rp = Rm * Rc * Rm.t();

    // ---- ② 姿态误差 ----
    // 姿态错 delta 会让位置按 |p| 为力臂转一下：dp = -[p]x * delta，
    // 所以 R_att = [p]x * Sigma_att * [p]x^T。这一项在远距离时主导。
    const double sa = cfg_.ekf.sigma_att_deg * M_PI / 180.0;
    cv::Mat Sig = cv::Mat::zeros(3, 3, CV_64F);
    Sig.at<double>(0, 0) = sa * sa;
    Sig.at<double>(1, 1) = sa * sa;
    Sig.at<double>(2, 2) = sa * sa * 0.25;  // 绕竖直轴更准一些

    cv::Mat Sk = cv::Mat::zeros(3, 3, CV_64F);
    Sk.at<double>(0, 1) = -d.p_world[2]; Sk.at<double>(0, 2) =  d.p_world[1];
    Sk.at<double>(1, 0) =  d.p_world[2]; Sk.at<double>(1, 2) = -d.p_world[0];
    Sk.at<double>(2, 0) = -d.p_world[1]; Sk.at<double>(2, 1) =  d.p_world[0];
    Rp += Sk * Sig * Sk.t();

    // ---- ③ 未建模项（自车平移等）给一个各向同性下限 ----
    const double fl = cfg_.ekf.sigma_world_floor;
    for (int i = 0; i < 3; ++i) Rp.at<double>(i, i) += fl * fl;

    if (R_pos_out) *R_pos_out = Rp.clone();

    // ---- ④ 朝向不确定度 ----
    // 理想值 = 角点噪声 / 板的像素宽度；实测远大于此，
    // 因为薄板的 PnP 朝向本身病态，且姿态误差会串进来。
    const double px_w = std::max(1.0, fx * W / Z);
    const double sig_yaw_floor = cfg_.ekf.sigma_yaw_deg * M_PI / 180.0;
    const double sig_yaw = std::max(spx * 20.0 / px_w, sig_yaw_floor);

    cv::Mat R = cv::Mat::zeros(kMeasDim, kMeasDim, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R.at<double>(i, j) = Rp.at<double>(i, j);
    R.at<double>(3, 3) = sig_yaw * sig_yaw;
    return R;
}

TrackResult Tracker::step(double dt, const std::vector<Detection>& dets,
                          const cv::Matx33d& R_wc, const cv::Mat& K) {
    TrackResult res;

    // ---------- 未初始化：等第一块目标车 -----------------
    if (!initialized_) {
        const Detection* pick = nullptr;
        for (const auto& d : dets) {
            if (d.number != target_num_) continue;
            if (d.reproj_err > cfg_.tracker.max_reproj_err) continue;
            if (d.range > cfg_.thresh.max_range_m) continue;
            if (!pick || d.reproj_err < pick->reproj_err) pick = &d;
        }
        if (!pick) return res;

        ekf_.init(pick->p_world, pick->yaw_world);
        is_large_ = pick->is_large;
        initialized_ = true;
        lost_dt_ = 0.0;
        anchor_miss_ = 0;
        phase_ = 0.0;

        res.tracking = true;
        res.detected = true;
        res.updated = true;
        res.x = ekf_.x().clone();
        res.P = ekf_.P().clone();
        res.is_large = is_large_;
        return res;
    }

    // ---------- 已初始化：预测 ----------
    ekf_.predict(dt);

    const double yaw_state = ekf_.at(6);

    // ---------- 匹配 ----------
    // 两条铁律：
    //  ① 新息必须「绝对」：wrap(量测yaw − (θ + phase))。
    //     phase 只在换板时改变，绝不能每帧从量测反推——否则滤波器永远只看到
    //     残差，状态会沿 90° 这个自由度匀速滑走（踩过这个坑）。
    //  ② 选哪块板当锚，看「正对度」。板越接近相切，投影宽度越小，
    //     PnP 的深度误差按 1/cos² 放大，板位置被抖成几十厘米，
    //     整个车体坐标跟着甩，推算出来的另外三块板就整体偏移。
    const int n = std::max(2, cfg_.tracker.armors_num);
    const double step_ang = 2.0 * M_PI / n;
    const double anchor_yaw = yaw_state + phase_;

    struct Cand {
        const Detection* d = nullptr;
        double dyaw = 0.0;   // 相对当前锚定板相位的角差（绝对）
        double phase = 0.0;  // 若以它为锚，对应的相位
        double m2 = 1e9;
        double dist = 1e9;
        cv::Mat R4;
    };

    Cand cur;  // 当前锚定板
    Cand alt;  // 正对度最好的另一块板（换板候选）
    int same_id_count = 0;

    for (const auto& d : dets) {
        if (d.number != target_num_) continue;                     // ① ID 门
        if (d.reproj_err > cfg_.tracker.max_reproj_err) continue;   // ② 粗差门
        if (d.range > cfg_.thresh.max_range_m) continue;
        ++same_id_count;

        cv::Mat Rpos;
        const cv::Mat R4 = buildMeasNoise(d, R_wc, K, &Rpos);

        const double dyaw = wrapAngle(d.yaw_world - anchor_yaw);

        if (std::fabs(dyaw) < cfg_.tracker.max_match_yaw_diff) {
            // ③ 朝向对得上 → 同一块板
            const double m2 = ekf_.mahalanobisPos(d.p_world, Rpos, phase_);
            const double dist = cv::norm(d.p_world - ekf_.predictedArmorPos(phase_));
            if (m2 < cur.m2) {
                cur.d = &d;
                cur.dyaw = dyaw;
                cur.phase = phase_;
                cur.m2 = m2;
                cur.dist = dist;
                cur.R4 = R4;
            }
        } else {
            // 另一块板：用它相对 θ 的相位槽标识，作为换板候选。
            // 选择依据不是距离，而是「正对度」——越正对，PnP 条件数越好。
            const double dtheta = wrapAngle(d.yaw_world - yaw_state);
            int j = static_cast<int>(std::lround(dtheta / step_ang));
            j = ((j % n) + n) % n;
            const double ph = j * step_ang;
            const double resid = wrapAngle(dtheta - ph);
            if (std::fabs(resid) > 0.35 * step_ang) continue;

            const double m2 = ekf_.mahalanobisPos(d.p_world, Rpos, ph);
            const double dist = cv::norm(d.p_world - ekf_.predictedArmorPos(ph));

            const bool better = (alt.d == nullptr) || (d.obliquity < alt.d->obliquity);
            if (better) {
                alt.d = &d;
                alt.dyaw = resid;
                alt.phase = ph;
                alt.m2 = m2;
                alt.dist = dist;
                alt.R4 = R4;
            }
        }
    }

    res.tracking = true;
    res.is_large = is_large_;

    auto usable = [&](const Cand& c) {
        return c.d != nullptr && c.m2 < cfg_.ekf.gate_pos &&
               c.dist < cfg_.tracker.max_match_distance;
    };

    // ---------- 换板判定 ----------
    bool do_switch = false;
    if (usable(alt)) {
        const double margin = cfg_.tracker.switch_margin_deg * M_PI / 180.0;
        if (!usable(cur)) {
            ++anchor_miss_;
            do_switch = (anchor_miss_ >= cfg_.tracker.switch_frames);
        } else if (alt.d->obliquity + margin < cur.d->obliquity) {
            // 新板明显更正对 → 立刻切，不等旧的消失
            do_switch = true;
        }
    } else if (!usable(cur)) {
        ++anchor_miss_;
    }

    if (do_switch) {
        phase_ = alt.phase;  // θ 完全不动：只换一个相位去描述同一组板
        anchor_miss_ = 0;
        res.switched = true;
        cur = alt;
    }

    res.phase_idx = static_cast<int>(std::lround(phase_ / step_ang)) % n;

    if (usable(cur)) {
        res.detected = true;

        const double z_yaw = yaw_state + phase_ + cur.dyaw;
        cv::Vec4d z(cur.d->p_world[0], cur.d->p_world[1], cur.d->p_world[2], z_yaw);

        double nis = 0.0;
        res.updated = ekf_.update(z, cur.R4, cfg_.ekf.nis_gate, &nis, phase_);
        res.nis = nis;

        if (res.updated) {
            is_large_ = cur.d->is_large;
            lost_dt_ = 0.0;
        }
    }

    // 半径钳位，防止发散
    const double r = ekf_.at(8);
    if (r < cfg_.tracker.radius_min) {
        ekf_.setStateValue(8, cfg_.tracker.radius_min);
    } else if (r > cfg_.tracker.radius_max) {
        ekf_.setStateValue(8, cfg_.tracker.radius_max);
    }

    // 长时间没有有效更新：丢掉滤波器，等目标重新出现时重新初始化
    if (!res.updated) lost_dt_ += dt;
    if (lost_dt_ > cfg_.tracker.lost_time_thres) {
        initialized_ = false;
        lost_dt_ = 0.0;
        anchor_miss_ = 0;
        phase_ = 0.0;
        res.tracking = false;
        return res;
    }

    static const bool dbg = (std::getenv("RM_TRACK_DEBUG") != nullptr);
    if (dbg) {
        std::fprintf(stderr, "[dbg] yaw=%5.2f r=%.3f same=%d miss=%d |",
                     ekf_.at(6), ekf_.at(8), same_id_count, anchor_miss_);
        if (cur.d) {
            std::fprintf(stderr,
                         " phase=%3.0f cur_ob=%4.1f alt_ob=%4.1f d=%.3f m2=%.2f -> %s\n",
                         phase_ * 180.0 / M_PI, cur.d->obliquity * 180.0 / M_PI,
                         alt.d ? alt.d->obliquity * 180.0 / M_PI : -1.0, cur.dist, cur.m2,
                         res.switched ? "SWITCH" : (res.updated ? "UP" : "REJ"));
            std::fprintf(stderr, "      n_cam=(%+.2f,%+.2f,%+.2f) n_world=(%+.2f,%+.2f,%+.2f)\n",
                         cur.d->n_cam[0], cur.d->n_cam[1], cur.d->n_cam[2],
                         cur.d->n_world[0], cur.d->n_world[1], cur.d->n_world[2]);
        } else {
            std::fprintf(stderr, " no candidate -> REJ\n");
        }
    }

    res.x = ekf_.x().clone();
    res.P = ekf_.P().clone();
    res.is_large = is_large_;
    return res;
}

}  // namespace rm
