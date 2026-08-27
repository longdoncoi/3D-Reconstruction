#pragma once
#include <opencv2/core.hpp>
#include <vector>

class PointCloudFilter {
public:
    // Statistical Outlier Removal (SOR)
    static void statisticalOutlier(std::vector<cv::Point3f> &pts,
                                   std::vector<cv::Vec3b> &cols,
                                   float meanK = 50.0f,
                                   float stdDevMulThresh = 1.0f);

    // Radius Outlier Removal (ROR)
    static void radiusOutlier(std::vector<cv::Point3f> &pts,
                              std::vector<cv::Vec3b> &cols,
                              float radius = 0.02f,
                              int minNeighbors = 3);

    static void adaptiveRadiusOutlier(std::vector<cv::Point3f> &pts,
                                      std::vector<cv::Vec3b> &cols,
                                      float radiusMultiplier = 3.0f,
                                      int minNeighbors = 2);

    // VoxelGrid downsampling (preserves color by nearest-point selection)
    static void voxelGrid(std::vector<cv::Point3f> &pts,
                          std::vector<cv::Vec3b> &cols,
                          float leafSize = 0.005f);

    // Density-based outlier removal (brute-force, for small clouds)
    static void densityFilter(std::vector<cv::Point3f> &pts,
                              std::vector<cv::Vec3b> &cols,
                              float radius, int minNeighbors);
};
