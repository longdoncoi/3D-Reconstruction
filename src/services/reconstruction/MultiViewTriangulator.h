#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "FeatureTrack.h"
#include "ReconstructionConfig.h" // CameraParams

class MultiViewTriangulator {
public:
    struct Params {
        double maxReprojError           = 2.5;  // px — nới từ 1.2 (quá chặt cho DLT đa-view)
        double minTriangulationAngleDeg = 1.0;  // độ — chỉ áp dụng khi track có đúng 2-view
        int    minObservations          = 2;    // ★ hạ từ 3 → 2
    };

    static bool triangulateTrack(FeatureTrack &track,
                                 const std::vector<CameraParams> &camParams,
                                 const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                                 const Params &params);

    static void refinePoint(FeatureTrack &track,
                            const std::vector<CameraParams> &camParams,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            int iterations = 5);

private:
    static double maxPairwiseAngleDeg(const FeatureTrack &track,
                                      const std::vector<CameraParams> &camParams);
};
