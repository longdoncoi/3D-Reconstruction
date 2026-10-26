#ifndef IAISERVICE_H
#define IAISERVICE_H

#include <QString>
#include <vector>
// ABI boundary: plugin must not link OpenCV just to include this interface.
namespace cv { class Mat; }

struct AIDetectionResult {
    int classId = -1;
    float confidence = 0.0f;
    int x = 0, y = 0, width = 0, height = 0; // bbox, OpenCV-independent
};

/// The outcome of a detection call: "nothing detected" and "the inference
/// failed" are different answers, so the caller gets a flag instead of an
/// empty vector that means both.
struct AIDetectionOutcome {
    bool ok = false;
    std::vector<AIDetectionResult> objects;
};

/**
 * @brief Interface trừu tượng cho dịch vụ AI (Detection & Segmentation).
 *
 * Plugin sử dụng interface này thay vì phụ thuộc trực tiếp vào AIProcessor.
 *
 * Cách dùng trong Plugin:
 *   auto* ai = ctx->services()->get<IAIService>();
 *   if (ai && ai->isDetectionReady()) {
 *       cv::Mat result = ai->runDetection(inputMat);
 *   }
 *
 * Model LOADING is deliberately not part of this interface: the loaders are
 * synchronous/blocking, and a GUI slot calling one would freeze the app. A
 * plugin triggers a load through IAIModelLoadService (AppCore owns the
 * worker) and observes is*Ready() here. The internal loader seam is
 * IAIModelAdmin in src/services/ai (not plugin-facing).
 */
class IAIService {
public:
    /// ServiceRegistry lookup key (see ServiceRegistry.h).
    static constexpr const char *ServiceKey =
        "com.3dreconstruction.service.IAIService";

    virtual ~IAIService() = default;

    // --- State Query ---
    /** Kiểm tra detection model đã sẵn sàng chưa. */
    virtual bool isDetectionReady() const = 0;

    /** Kiểm tra segmentation model đã sẵn sàng chưa. */
    virtual bool isSegmentationReady() const = 0;

    /** Kiểm tra tracking model đã sẵn sàng chưa. */
    virtual bool isTrackingReady() const = 0;

    // --- Inference ---
    /**
     * Chạy object detection và trả về danh sách bounding boxes có cấu trúc.
     *
     * `ok` is what makes the result unambiguous: an empty `objects` list with
     * `ok == true` means "the inference ran and found nothing", while
     * `ok == false` means the model is not loaded or the inference failed. A
     * bare `std::vector` conflated the two, so a caller could not tell a
     * clean frame from a broken backend.
     */
    virtual AIDetectionOutcome detectObjects(const cv::Mat& inputImage) = 0;

    /**
     * Chạy object detection trên ảnh đầu vào.
     * @return cv::Mat với bounding boxes đã vẽ, hoặc empty nếu lỗi.
     */
    virtual cv::Mat runDetection(const cv::Mat& inputImage) = 0;

    /**
     * Chạy instance segmentation trên ảnh đầu vào.
     * @return cv::Mat với masks đã vẽ, hoặc empty nếu lỗi.
     */
    virtual cv::Mat runSegmentation(const cv::Mat& inputImage) = 0;

    /**
     * Chạy object tracking trên frame đầu vào (có lưu trạng thái).
     * @return cv::Mat với bounding boxes và tracking IDs đã vẽ.
     */
    virtual cv::Mat runTracking(const cv::Mat& inputImage) = 0;

    /**
     * Reset trạng thái tracking (ví dụ khi chuyển video mới).
     */
    virtual void resetTrackingState() = 0;

    /**
     * Đặt ngưỡng tin cậy (confidence threshold) cho detection/segmentation.
     * Giá trị từ 0.0 đến 1.0, mặc định 0.5.
     */
    virtual void setConfidenceThreshold(float threshold) = 0;
    virtual float confidenceThreshold() const = 0;
};

#endif // IAISERVICE_H
