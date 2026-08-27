#ifndef RECONSTRUCTION_CONFIG_H
#define RECONSTRUCTION_CONFIG_H

#include <opencv2/core.hpp>
#include <QString>

struct CameraParams {
    QString imageName;
    cv::Mat K; // 3x3
    cv::Mat R; // 3x3
    cv::Mat t; // 3x1
    cv::Mat P; // 3x4 = K * [R|t]
};

struct SiftConfig {
    int nfeatures = 20000;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.005;
    double edgeThreshold = 12.0;
    double sigma = 1.6;
};

struct MatchingConfig {
    float ratioThreshold = 0.78f;
    bool crossCheck = true;
};

struct FilterConfig {
    // Statistical Outlier Removal
    int sorMeanK = 50;
    float sorStdDevMul = 1.4f;

    // Radius Outlier Removal
    float rorRadius = 0.006f;
    int rorMinNeighbors = 2;

    // Voxel Grid
    float voxelLeafSize = 0.00025f;

    // ★ MỚI — Profile "track-based / ground-truth" — cloud đã dedup + validate đa-view,
    // không cần lọc mật độ gắt vì mỗi điểm đã qua kiểm tra reprojection + parallax angle
    int   sorMeanKTrack        = 15;
    float sorStdDevMulTrack    = 2.5f;
    float rorRadiusTrack       = 0.015f;
    int   rorMinNeighborsTrack = 2;
    float voxelLeafSizeTrack   = 0.0002f;
};

struct ReconstructionConfig {
    SiftConfig sift;
    MatchingConfig matching;
    FilterConfig filter;
    
    // Global limits
    double reprojectionErrorMax = 1.5;
    int minMatches = 20;
    int searchWindow = 15;
    int minTrackObservations = 2;
    double pairFundamentalRansacThreshold = 1.5;
};

#endif // RECONSTRUCTION_CONFIG_H
