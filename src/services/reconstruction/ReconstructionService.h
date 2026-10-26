#ifndef RECONSTRUCTIONSERVICE_H
#define RECONSTRUCTIONSERVICE_H

#include "IReconstructionService.h"
#include "ReconstructionPipeline.h"
#include "Global.h"
#include <QThread>
#include <memory>
#include <atomic>
#include <mutex>
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

protected:
    /// Test seam: injects the pipeline so the failure path of a run (a
    /// snapshot conversion that throws) can be exercised without a real,
    /// multi-gigabyte reconstruction. Production uses the default ctor.
    explicit ReconstructionService(std::unique_ptr<ReconstructionPipeline> pipeline);

private:
    /// Teardown sau khi worker phát finished(): join thread, xóa worker/thread
    /// theo ownership tường minh (không dùng auto-delete → không có race giữa
    /// lúc thread vừa finished và lúc destructor kiểm tra trạng thái).
    void finishRun(bool ok);

    /// Builds the published snapshot + point count from the pipeline and
    /// publishes them. The caller MUST hold m_mutex (the pipeline lock) —
    /// the pipeline data is read under it. The publication itself takes
    /// m_publishedMutex, so the lock order stays m_mutex → m_publishedMutex.
    ///
    /// Returns false when the snapshot could not be built (the conversion
    /// allocates, and ONNX/OpenCV/the allocator all throw). The failure is
    /// reported through m_lastError; it never escapes the caller — both call
    /// sites (the worker thread and processPointCloud on the GUI thread) must
    /// survive it. A run whose result could not be published is a FAILED run.
    bool publishResult();

    std::unique_ptr<ReconstructionPipeline> m_pipeline;
    std::atomic<bool> m_running{false};
    /// Yêu cầu hủy cooperative — set bởi stopReconstruction()/dtor, đọc bởi
    /// predicate truyền vào pipeline. Reset ở đầu mỗi lần reconstruct().
    std::atomic<bool> m_cancelRequested{false};
    /// Bảo vệ chính pipeline: worker giữ khoá độc quyền suốt quá trình
    /// reconstruct, nên MỌI lời gọi vào m_pipeline đều phải đi qua khoá này.
    mutable std::shared_mutex m_mutex;
    /// Bảo vệ state đã "publish" cho UI (m_imageList + m_snapshot +
    /// m_publishedPointCount), TÁCH RIÊNG khỏi m_mutex: worker giữ m_mutex suốt
    /// quá trình chạy, nên một truy vấn UI đọc bản publish phải không bao giờ
    /// phải chờ nó (nếu dùng chung khoá thì UI đóng băng suốt thời gian
    /// reconstruct).
    ///
    /// Đây là bản publish DUY NHẤT cho UI: cả kết quả lẫn số điểm đều được
    /// worker publish khi lần chạy kết thúc, nên UI không bao giờ phải chạm
    /// vào pipeline — kể cả khi đang reconstruct.
    ///
    /// Thứ tự khoá bất biến: m_mutex TRƯỚC m_publishedMutex, không bao giờ
    /// ngược lại (xem publishResult / setImages).
    mutable std::mutex m_publishedMutex;
    /// The image list, published when the images are set. The UI reads this
    /// copy instead of the pipeline's.
    QStringList m_imageList;
    /// The published result. Rebuilt per run and replaced (never mutated in
    /// place) so callers holding a snapshot stay valid.
    std::shared_ptr<const PointCloudSnapshot> m_snapshot;
    /// The number of points in the published snapshot — published together
    /// with it, so the UI never has to touch the pipeline to report it.
    size_t m_publishedPointCount = 0;
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
