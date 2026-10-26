#ifndef RECONSTRUCTION_PIPELINE_H
#define RECONSTRUCTION_PIPELINE_H

#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <functional>
#include <memory>
#include <vector>

#include "Global.h"
#include "PointCloudData.h"
#include "ReconstructionConfig.h"

/**
 * Orchestrator of the 3D reconstruction pipeline.
 *
 * Owns all storage (images, features, camera model, result cloud) and only
 * sequences the stages; every algorithm lives in its own translation unit:
 *
 *   - feature extraction : FeatureExtractor (parallel per image)
 *   - reconstruction     : TrackSfM (ground-truth poses) OR IncrementalSfM
 *                          (estimated poses), picked from the loaded params
 *   - cleanup            : ReconstructionPostProcessor::processPointCloud
 *   - densify / texture / Poisson meshing : ReconstructionPostProcessor
 *   - disk cache (PLY)   : private helpers below
 *
 * PCL types no longer leak into this header: cache/mesh I/O is confined to
 * the .cpp translation unit.
 */
class APP_EXPORT ReconstructionPipeline {
public:
    ReconstructionPipeline();
    virtual ~ReconstructionPipeline();

    void setImages(const std::vector<QString> &imagePaths);
    bool loadCameraParams(const QString &paramsFilePath);

    /**
     * Chạy toàn bộ pipeline (đồng bộ, blocking).
     *
     * @param shouldCancel  Predicate hủy cooperative, được kiểm tại các ranh
     *                      giới stage (entry / từng ảnh / giữa các bước hậu kỳ).
     *                      Rỗng = không hỗ trợ hủy. Khi hủy, trả về false và
     *                      errorString() = "Reconstruction cancelled.".
     * @return true nếu có point cloud kết quả.
     */
    /// Virtual so the service's failure paths (a snapshot conversion that
    /// throws) can be exercised with a stub instead of a multi-gigabyte run.
    virtual bool reconstruct(const std::function<bool()> &shouldCancel = {});

    /// Lý do thất bại của lần reconstruct() gần nhất (rỗng nếu thành công).
    /// Dùng để hiển thị chẩn đoán cho người dùng thay vì thông báo chung chung.
    QString errorString() const { return m_lastError; }

    // Configuration access
    ReconstructionConfig& config() { return m_config; }
    void setConfig(const ReconstructionConfig &cfg);

    const std::vector<cv::Point3f>& getPointCloud() const;
    const std::vector<cv::Vec3b>& getPointColors() const;
    size_t pointCount() const { return points3D.size(); }
    virtual bool hasResult() const { return !points3D.empty(); }
    std::vector<QString> getImages() const { return imageFiles; }

    /// Builds the ABI-neutral snapshot of the current point cloud.
    ///
    /// The service publishes it so the UI never has to touch the pipeline
    /// (which the worker holds locked for the whole reconstruction). The
    /// conversion lives HERE, next to the data it converts, so it can be
    /// measured by the fast pipeline tests instead of only by a completed
    /// reconstruction.
    virtual std::shared_ptr<const PointCloudSnapshot> buildSnapshot() const;

    /// Pipeline lọc tổng hợp (adaptive ROR → SOR → voxel) trên cloud hiện có.
    /// Delegate sang ReconstructionPostProcessor (không còn nội bộ ở đây).
    void processPointCloud();

private:
    ReconstructionConfig m_config;

    // Dữ liệu
    std::vector<QString> imageFiles;
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;

    cv::Mat K_fallback; // intrinsic khi không có params
    cv::Mat distCoeffs; // (bỏ qua distortion)

    // Ground truth params
    std::vector<CameraParams> camParams;
    bool hasGroundTruthParams = false;
    /// Whether camParams has been matched to the loaded images BY NAME. The
    /// ground-truth branch may only run when this is true: a positional
    /// binding would assign every pose to the wrong image.
    bool cameraParamsMatched = false;

    /// Bind the ground-truth camera parameters to the loaded images.
    ///
    /// Returns false when the names do not match (or the file describes
    /// fewer images than were loaded); the caller then falls back to the
    /// self-estimating branch instead of using poses that belong to other
    /// images.
    bool matchCameraParamsToImages();

    // Kết quả
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> colors;

    // Lý do thất bại gần nhất (xem errorString())
    QString m_lastError;

    // Predicate hủy cooperative — sống trong đúng một lần reconstruct().
    // Wrapper reconstruct() set trước / clear sau reconstructImpl().
    std::function<bool()> m_shouldCancel;

    /// true nếu caller đã yêu cầu hủy kể từ checkpoint gần nhất.
    bool cancelRequested() const { return m_shouldCancel && m_shouldCancel(); }

    /// Checkpoint hủy: set m_lastError + trả true nếu đã yêu cầu hủy.
    bool abortIfCancelled();

    // Các bước của reconstruct() — tách ra để orchestrator dễ đọc,
    // mỗi bước xử lý đúng một việc.
    bool reconstructImpl();
    void extractFeatures(int idx);
    void computeCachePaths(QString &cachePath, QString &meshCachePath) const;
    bool loadFromCache(const QString &cachePath, const QString &meshCachePath);
    bool extractAllFeatures();
    bool runSfMBranch();
    void finalizeReconstruction(const QString &cachePath,
                                const QString &meshCachePath);
};

#endif
