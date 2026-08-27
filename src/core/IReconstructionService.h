#ifndef IRECONSTRUCTIONSERVICE_H
#define IRECONSTRUCTIONSERVICE_H

#include <QString>
#include <QStringList>
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Interface trừu tượng cho dịch vụ tái tạo 3D.
 *
 * Plugin sử dụng interface này thay vì phụ thuộc trực tiếp vào
 * ReconstructionPipeline (concrete class).
 *
 * Cách dùng trong Plugin:
 *   auto* svc = ctx->services()->get<IReconstructionService>();
 *   if (svc) svc->setImages(paths);
 */
class IReconstructionService {
public:
    virtual ~IReconstructionService() = default;

    // --- Input ---
    /** Đặt danh sách ảnh đầu vào cho quá trình tái tạo. */
    virtual void setImages(const QStringList& paths) = 0;

    /** Tải camera parameters từ file. Trả về true nếu thành công. */
    virtual bool loadCameraParams(const QString& filePath) = 0;

    /** Cấu hình các tham số tái tạo từ Settings (mật độ, tính năng, lọc nhiễu). */
    virtual void configure(int densityLevel, int maxFeatures, bool autoClean) = 0;

    // --- Execution ---
    /** Bắt đầu quá trình tái tạo (blocking hoặc async tùy impl). */
    virtual bool reconstruct() = 0;

    /** Kiểm tra có đang chạy không (dùng cho async impl). */
    virtual bool isRunning() const = 0;

    /** Dừng quá trình tái tạo nếu đang chạy. */
    virtual void stopReconstruction() = 0;

    // --- Output / Query ---
    /** Lấy danh sách đường dẫn ảnh đã set. */
    virtual QStringList getImageList() const = 0;

    /** Lấy point cloud kết quả (3D points). */
    virtual std::vector<cv::Point3f> getPointCloud() const = 0;

    /** Lấy màu của từng point cloud point. */
    virtual std::vector<cv::Vec3b> getPointColors() const = 0;

    /** Kiểm tra có kết quả hợp lệ không. */
    virtual bool hasResult() const = 0;

    /** Lọc và xử lý point cloud sau khi reconstruct. */
    virtual void processPointCloud() = 0;
};

#endif // IRECONSTRUCTIONSERVICE_H
