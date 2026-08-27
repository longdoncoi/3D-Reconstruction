#include "AIYoloPostProcessor.h"

#include <algorithm>
#include <numeric>

namespace AIYoloPostProcessor {

YoloDetections parsePredictions(const cv::Mat &predictions,
                                cv::Size imageSize,
                                cv::Size modelSize,
                                float scoreThreshold,
                                int maskCoefficientCount)
{
    YoloDetections detections;
    const int numClasses = predictions.cols - 4 - maskCoefficientCount;
    const float xFactor = imageSize.width / static_cast<float>(modelSize.width);
    const float yFactor = imageSize.height / static_cast<float>(modelSize.height);

    for (int row = 0; row < predictions.rows; ++row) {
        const float *data = predictions.ptr<float>(row);

        float maxScore = -1.0f;
        int classId = -1;
        for (int classIndex = 0; classIndex < numClasses; ++classIndex) {
            if (data[4 + classIndex] > maxScore) {
                maxScore = data[4 + classIndex];
                classId = classIndex;
            }
        }

        if (maxScore < scoreThreshold) continue;

        const float centerX = data[0];
        const float centerY = data[1];
        const float width = data[2];
        const float height = data[3];

        detections.classIds.push_back(classId);
        detections.confidences.push_back(maxScore);
        detections.boxes.push_back(cv::Rect(
            static_cast<int>((centerX - 0.5f * width) * xFactor),
            static_cast<int>((centerY - 0.5f * height) * yFactor),
            static_cast<int>(width * xFactor),
            static_cast<int>(height * yFactor)));

        if (maskCoefficientCount > 0) {
            detections.maskCoefficients.emplace_back(
                data + 4 + numClasses,
                data + 4 + numClasses + maskCoefficientCount);
        }
    }

    return detections;
}

std::vector<int> applyNms(const std::vector<cv::Rect> &boxes,
                          const std::vector<float> &confidences,
                          float nmsThreshold)
{
    std::vector<int> keptIndices;
    if (boxes.empty()) return keptIndices;

    std::vector<int> sortedIndices(boxes.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&](int left, int right) { return confidences[left] > confidences[right]; });

    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t sortedIndex = 0; sortedIndex < sortedIndices.size(); ++sortedIndex) {
        const int index = sortedIndices[sortedIndex];
        if (suppressed[index]) continue;

        keptIndices.push_back(index);

        for (size_t candidateIndex = sortedIndex + 1; candidateIndex < sortedIndices.size(); ++candidateIndex) {
            const int candidate = sortedIndices[candidateIndex];
            if (suppressed[candidate]) continue;

            const cv::Rect intersection = boxes[index] & boxes[candidate];
            const float intersectionArea = static_cast<float>(intersection.area());
            const float unionArea = static_cast<float>(boxes[index].area() + boxes[candidate].area()) - intersectionArea;
            const float iou = unionArea > 0.0f ? intersectionArea / unionArea : 0.0f;

            if (iou > nmsThreshold) {
                suppressed[candidate] = true;
            }
        }
    }

    return keptIndices;
}

}
