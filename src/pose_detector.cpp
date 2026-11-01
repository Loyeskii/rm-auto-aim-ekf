#include "pose_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace rm {

namespace {

// 两个轴对齐矩形的 IoU，用于 NMS
float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.f, x2 - x1);
    const float h = std::max(0.f, y2 - y1);
    const float inter = w * h;
    const float uni = a.width * a.height + b.width * b.height - inter;
    return (uni <= 0.f) ? 0.f : inter / uni;
}

// 四个角点的外接正矩形
cv::Rect2f bboxOf(const cv::Point2f c[4]) {
    float x0 = c[0].x, x1 = c[0].x, y0 = c[0].y, y1 = c[0].y;
    for (int i = 1; i < 4; ++i) {
        x0 = std::min(x0, c[i].x); x1 = std::max(x1, c[i].x);
        y0 = std::min(y0, c[i].y); y1 = std::max(y1, c[i].y);
    }
    return cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
}

}  // namespace

PoseDetector::PoseDetector(const Config& cfg) : cfg_(cfg) {
    try {
        net_ = cv::dnn::readNetFromONNX(cfg_.detect.model);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[info] 装甲板检测网络已加载: " << cfg_.detect.model << "\n";
    } catch (const cv::Exception& e) {
        err_ = e.what();
        net_ = cv::dnn::Net();
        std::cerr << "[warn] 检测网络加载失败: " << cfg_.detect.model << "\n"
                  << "       " << err_ << "\n";
    }
}

// ------------------------------------------------------------
// 预处理：原图 → 4 维 blob
//   1) resize 到 640×480：素材 960×720 与 640×480 同为 4:3，
//      直接拉伸即可，不需要 letterbox，也就省掉了 padding 偏移的反算。
//   2) BGR → RGB：网络训练时是 RGB 输入。
//   3) 归一化到 0~1，HWC → CHW。
// ------------------------------------------------------------
cv::Mat PoseDetector::forward(const cv::Mat& bgr) {
    cv::Mat blob = cv::dnn::blobFromImage(
        bgr, 1.0 / 255.0,
        cv::Size(cfg_.detect.input_w, cfg_.detect.input_h),
        cv::Scalar(), cfg_.detect.swap_rb, /*crop=*/false);
    net_.setInput(blob);
    return net_.forward();     // [1, 21, 6300]
}

std::vector<Armor> PoseDetector::detect(const cv::Mat& bgr) {
    std::vector<Armor> out;
    if (net_.empty() || bgr.empty()) return out;

    cv::Mat raw = forward(bgr);            // [1, 21, 6300]
    if (raw.empty()) return out;

    // ---- 1) 布局转换：[1, 21, 6300] → [6300, 21] ----
    // 网络是"通道优先"布局，也就是每一行是一个属性、每一列是一个候选；
    // 解码时要反过来，一行读一个候选的 21 个属性。
    const int rows = raw.size[1];          // 21
    const int cols = raw.size[2];          // 6300
    cv::Mat m = raw.reshape(1, rows);      // 21 x 6300
    cv::Mat mt;
    cv::transpose(m, mt);                  // 6300 x 21

    // ---- 2) 坐标还原比例 ----
    // 输出坐标是在 640×480 的输入尺度下的像素，要映射回原图。
    const float sx = static_cast<float>(bgr.cols) / cfg_.detect.input_w;
    const float sy = static_cast<float>(bgr.rows) / cfg_.detect.input_h;

    struct Cand { Armor a; float score; cv::Rect2f box; };
    std::vector<Cand> cands;
    cands.reserve(64);

    for (int i = 0; i < cols; ++i) {
        const float* p = mt.ptr<float>(i);

        // ---- 3) 类别：取 9 个分数里的最大值 ----
        // 第 4~12 列是类别分数（该模型输出的就是概率，无需再 sigmoid）
        int best_cls = -1;
        float best_score = 0.f;
        for (int k = 0; k < 9; ++k) {
            if (p[4 + k] > best_score) { best_score = p[4 + k]; best_cls = k; }
        }
        if (best_score < static_cast<float>(cfg_.detect.conf_thres)) continue;

        // ---- 4) 关键点：第 13~20 列是 4 个点 × (x, y) ----
        // 实测该模型的点序是「左上、左下、右下、右上」，
        // 而 PnP、画框、透视变换都要求「左上、右上、右下、左下」，
        // 所以这里必须重排：0→TL, 3→TR, 2→BR, 1→BL。
        cv::Point2f kp[4];
        for (int k = 0; k < 4; ++k) {
            kp[k] = cv::Point2f(p[13 + 2 * k] * sx, p[14 + 2 * k] * sy);
        }

        Armor a;
        a.corners[0] = kp[0];   // 左上
        a.corners[1] = kp[3];   // 右上
        a.corners[2] = kp[2];   // 右下
        a.corners[3] = kp[1];   // 左下
        a.number = best_cls;
        a.digit_conf = best_score;
        a.color = ArmorColor::UNKNOWN;   // 本方案不做颜色判定

        // ---- 5) 大/小装甲板：用框的宽高比判断 ----
        // 小装甲板 135×125≈1.08，大装甲板 225×125≈1.8。
        // 注意不能读网络的 box（实测 w/h 恒为 0），只能自己算。
        const float qw = static_cast<float>(cv::norm(a.corners[1] - a.corners[0]));
        const float qh = static_cast<float>(cv::norm(a.corners[3] - a.corners[0]));
        if (qh < 1e-3f || qw < 1e-3f) continue;
        a.is_large = (qw / qh) > static_cast<float>(cfg_.detect.large_aspect);

        {
            static const bool dbg = (std::getenv("RM_TRACK_DEBUG") != nullptr);
            if (dbg) {
                std::fprintf(stderr, "[det] cls=%d conf=%.2f asp=%.2f qw=%.0f qh=%.0f -> %s\n",
                             best_cls, best_score, qw / qh, qw, qh,
                             a.is_large ? "LARGE" : "small");
            }
        }

        cands.push_back({a, best_score, bboxOf(a.corners)});
    }

    // ---- 6) NMS：网络导出时 nms=False，必须自己做 ----
    // 同一块板常被预测多次（实测同一位置会出现 2~3 个几乎重合的候选）。
    std::sort(cands.begin(), cands.end(),
              [](const Cand& x, const Cand& y) { return x.score > y.score; });

    std::vector<int> keep;
    for (size_t i = 0; i < cands.size(); ++i) {
        bool suppressed = false;
        for (int k : keep) {
            if (iou(cands[i].box, cands[k].box) > static_cast<float>(cfg_.detect.nms_thres)) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        keep.push_back(static_cast<int>(i));
        if (static_cast<int>(keep.size()) >= cfg_.detect.max_dets) break;
    }

    out.reserve(keep.size());
    for (int k : keep) out.push_back(cands[static_cast<size_t>(k)].a);
    return out;
}

}  // namespace rm
