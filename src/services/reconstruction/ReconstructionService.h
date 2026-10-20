#ifndef RECONSTRUCTIONSERVICE_H
#define RECONSTRUCTIONSERVICE_H

#include "IReconstructionService.h"
#include "ReconstructionPipeline.h"
#include "Global.h"
#include <QThread>
#include <memory>
#include <atomic>
#include <shared_mutex>

/**
 * @brief Concrete implementation của IReconstructionService.
 *
 * Wrap ReconstructionPipeline, expose interface sạch cho plugin sử dụng.
 * Plugin không cần include ReconstructionPipeline.h và không tự quản lý
 * thread: reconstruct() chạy worker nội bộ và phát reconstructFinished().
 */
class APP_EXPORT ReconstructionService : public IReconstructionService {
public:
    ReconstructionService();
    ~ReconstructionService() override;

    // IReconstructionService
    void setImages(const QStringList& paths) override;
    bool loadCameraParams(const QString& filePath) override;
    void configure(int densityLevel, int maxFeatures, bool autoClean) override;
    bool reconstruct() override;
    bool isRunning() const override;
    void stopReconstruction() override;
    QStringList getImageList() const override;
    std::shared_ptr<const PointCloudSnapshot> pointCloudSnapshot() const override;
    size_t pointCount() const override;
    bool hasResult() const override;
    void processPointCloud() override;

    /// Synchronous execution core — chỉ dành cho worker nội bộ
    /// (ReconstructionWorker, chạy trên m_thread). Bên ngoài hãy gọi
    /// reconstruct().
    bool runPipeline();

private:
    /// Teardown sau khi worker phát finished(): join thread, xóa worker/thread
    /// theo ownership tường minh (không dùng auto-delete → không có race giữa
    /// lúc thread vừa finished và lúc destructor kiểm tra trạng thái).
    void finishRun(bool ok);

    std::unique_ptr<ReconstructionPipeline> m_pipeline;
    std::atomic<bool> m_running{false};
    /// Yêu cầu hủy cooperative — set bởi stopReconstruction()/dtor, đọc bởi
    /// predicate truyền vào pipeline. Reset ở đầu mỗi lần reconstruct().
    std::atomic<bool> m_cancelRequested{false};
    mutable std::shared_mutex m_mutex;
    /// Cached ABI-neutral result. Built once per completed run and replaced
    /// (never mutated in place) so callers holding a snapshot stay valid.
    mutable std::shared_ptr<const PointCloudSnapshot> m_snapshot;
    /// QThread nội bộ chạy ReconstructionWorker (moveToThread pattern);
    /// không bao giờ move `this` — chỉ worker được move. Ownership tường minh:
    /// service xóa cả hai trong finishRun()/destructor sau khi join thread.
    QThread* m_thread = nullptr;
    /// Worker sống trong m_thread; raw ptr để destructor join-rồi-xóa khi
    /// service chết giữa chừng.
    class ReconstructionWorker* m_worker = nullptr;
    /// Lý do thất bại của lần chạy pipeline gần nhất (đọc sau khi worker xong).
    QString m_lastError;
};

#endif // RECONSTRUCTIONSERVICE_H
