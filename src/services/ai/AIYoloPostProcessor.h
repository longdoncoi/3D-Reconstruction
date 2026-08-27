#ifndef AI_YOLO_POST_PROCESSOR_H
#define AI_YOLO_POST_PROCESSOR_H

#include <opencv2/core.hpp>
#include <vector>

struct YoloDetections {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> maskCoefficients;
};

namespace AIYoloPostProcessor {

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
