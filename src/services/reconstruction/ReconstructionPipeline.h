#ifndef RECONSTRUCTION_PIPELINE_H
#define RECONSTRUCTION_PIPELINE_H

#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h> // nếu muốn lưu kết quả
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

#include "Global.h"
#include "ReconstructionConfig.h"
#include "AppConstants.h"
#include "FeatureTrack.h"
#include "MultiViewTriangulator.h"

class APP_EXPORT ReconstructionPipeline {
public:
    ReconstructionPipeline();
    ~ReconstructionPipeline();

    void setImages(const std::vector<QString> &imagePaths);
    bool loadCameraParams(const QString &paramsFilePath);
    bool reconstruct();
    
    // Configuration access
    ReconstructionConfig& config() { return m_config; }
    void setConfig(const ReconstructionConfig &cfg);

    std::vector<cv::Point3f> getPointCloud() const;
    std::vector<cv::Vec3b> getPointColors() const;
    std::vector<QString> getImages() const { return imageFiles; }

    // Lọc Statistical Outlier Removal (SOR)
    void statisticalOutlierFilter(float meanK = 50, float stdDevMulThresh = 1.0);
    // Lọc Radius Outlier Removal (ROR)
    void radiusOutlierFilter(float radius = 0.02, int minNeighbors = 3);
    // Giảm mẫu bằng Voxel Grid
    void voxelGridDownsample(float leafSize = 0.005);

    // Pipeline lọc tổng hợp
    void processPointCloud();

private:
    ReconstructionConfig m_config;
    
    // Dữ liệu
    std::vector<QString> imageFiles;
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    cv::Ptr<cv::Feature2D> detector;

    cv::Mat K_fallback; // intrinsic khi không có params
    cv::Mat distCoeffs; // (bỏ qua distortion)

    // Ground truth params
    std::vector<CameraParams> camParams;
    bool hasGroundTruthParams = false;
    bool m_usedTrackBasedGroundTruth = false;

    // Kết quả
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> colors;

    // Hàm nội bộ
    void extractFeatures(int idx);
    void matchFeatures(int idx1, int idx2, std::vector<cv::DMatch> &goodMatches);
    bool estimatePoseFromMatches(const std::vector<cv::Point2f> &pts1,
                                 const std::vector<cv::Point2f> &pts2, cv::Mat &R,
                                 cv::Mat &t);
    void doTriangulate(const cv::Mat &P0, const cv::Mat &P1,
                       const std::vector<cv::Point2f> &pts0,
                       const std::vector<cv::Point2f> &pts1,
                       std::vector<cv::Point3f> &outPts);
    double computeReprojectionError(const cv::Mat &P, const cv::Point3f &pt3d,
                                    const cv::Point2f &pt2d);

    // Hai phương thức reconstruct riêng
    bool reconstructWithGroundTruth();
    bool reconstructWithEstimatedPose();

    PointCloudT::Ptr convertToPCLCloud(const std::vector<cv::Point3f> &pts,
                                       const std::vector<cv::Vec3b> &cols) const;
    void convertFromPCLCloud(PointCloudT::Ptr cloud,
                             std::vector<cv::Point3f> &pts,
                             std::vector<cv::Vec3b> &cols);
};

#endif
