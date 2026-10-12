#ifndef AI_YOLO_POST_PROCESSOR_H
#define AI_YOLO_POST_PROCESSOR_H

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

struct YoloDetections {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> maskCoefficients;
};

namespace AIYoloPostProcessor {

/// Square input size assumed when a model exposes a dynamic input shape.
///
/// This is the value every shipped export is trained at; it only acts as a
/// fallback — a session that reports a concrete shape is driven with that
/// shape instead, so `--imgsz` at export time is honoured at inference time.
constexpr int kDefaultModelInputSize = 640;

/// Derive the ``(width, height)`` a session expects from its NCHW input shape.
///
/// ``shape`` is the ONNX input shape as ONNX Runtime reports it — ``[N, C, H,
/// W]`` for every model this project exports. A dynamic axis (negative), a
/// non-positive extent or an unexpected rank falls back to
/// :data:`kDefaultModelInputSize` rather than producing a zero-sized tensor.
cv::Size modelInputSize(const std::vector<int64_t> &shape);

YoloDetections parsePredictions(const cv::Mat &predictions,
                                cv::Size imageSize,
                                cv::Size modelSize,
                                float scoreThreshold,
                                int maskCoefficientCount = 0);

std::vector<int> applyNms(const std::vector<cv::Rect> &boxes,
                          const std::vector<float> &confidences,
                          float nmsThreshold);

}

#endif // AI_YOLO_POST_PROCESSOR_H
