#include "imu_reader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace rm {

Quat slerp(const Quat& a_in, const Quat& b_in, double u) {
    Quat a = a_in, b = b_in;
    double d = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;

    // 四元数 q 与 -q 表示同一旋转，取同侧再插值，否则会绕远路
    if (d < 0.0) {
        b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z;
        d = -d;
    }

    Quat out;
    if (d > 0.9995) {  // 夹角极小，线性插值后归一化，避免除零
        out.w = a.w + u * (b.w - a.w);
        out.x = a.x + u * (b.x - a.x);
        out.y = a.y + u * (b.y - a.y);
        out.z = a.z + u * (b.z - a.z);
    } else {
        const double th = std::acos(std::max(-1.0, std::min(1.0, d)));
        const double s = std::sin(th);
        const double wa = std::sin((1.0 - u) * th) / s;
        const double wb = std::sin(u * th) / s;
        out.w = wa * a.w + wb * b.w;
        out.x = wa * a.x + wb * b.x;
        out.y = wa * a.y + wb * b.y;
        out.z = wa * a.z + wb * b.z;
    }

    const double n = std::sqrt(out.w * out.w + out.x * out.x + out.y * out.y + out.z * out.z);
    if (n > 1e-12) {
        out.w /= n; out.x /= n; out.y /= n; out.z /= n;
    }
    return out;
}

bool ImuReader::load(const std::string& path, std::string* err) {
    t_.clear();
    q_.clear();

    std::ifstream ifs(path);
    if (!ifs) {
        if (err) *err = "打不开 IMU 文件: " + path;
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        double t, w, x, y, z;
        if (!(iss >> t >> w >> x >> y >> z)) continue;  // 跳过空行/脏行
        t_.push_back(t);
        q_.push_back(Quat{w, x, y, z});
    }

    if (t_.empty()) {
        if (err) *err = "IMU 文件里没有有效数据行: " + path;
        return false;
    }
    if (err) err->clear();
    return true;
}

Quat ImuReader::attitudeForFrame(size_t i, double latency) const {
    if (q_.empty()) return Quat{};
    if (i >= t_.size()) i = t_.size() - 1;

    const double tq = t_[i] - latency;

    // 落在首行之前 / 末行之后：直接取端点
    if (tq <= t_.front()) return q_.front();
    if (tq >= t_.back()) return q_.back();

    // 二分找到第一个 >= tq 的行
    const auto it = std::lower_bound(t_.begin(), t_.end(), tq);
    const size_t k = static_cast<size_t>(it - t_.begin());
    if (k == 0) return q_.front();

    const double t0 = t_[k - 1], t1 = t_[k];
    const double denom = t1 - t0;
    const double u = (denom > 1e-12) ? (tq - t0) / denom : 0.0;
    return slerp(q_[k - 1], q_[k], u);
}

double ImuReader::dtOfFrame(size_t i) const {
    if (t_.size() < 2) return 1.0 / 30.0;
    if (i == 0 || i >= t_.size()) return t_[1] - t_[0];
    return t_[i] - t_[i - 1];
}

cv::Matx33d ImuReader::rotationFromQuat(const Quat& q) {
    const double w = q.w, x = q.x, y = q.y, z = q.z;
    return cv::Matx33d(
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y));
}

cv::Matx33d ImuReader::cameraToBody() {
    // 列 = 相机三轴在机体系下的坐标
    return cv::Matx33d(0.0, 0.0, 1.0,
                       -1.0, 0.0, 0.0,
                       0.0, -1.0, 0.0);
}

}  // namespace rm
