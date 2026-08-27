#ifndef IAISERVICE_H
#define IAISERVICE_H

#include <QString>

namespace cv { class Mat; }

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
 */
class IAIService {
public:
    virtual ~IAIService() = default;

    // --- Model Loading ---
    /** Tải model detection (ONNX). Trả về true nếu thành công. */
    virtual bool loadDetectionModel(const QString& modelPath) = 0;

    /** Tải model segmentation (ONNX). Trả về true nếu thành công. */
    virtual bool loadSegmentationModel(const QString& modelPath) = 0;

    /** Tải model tracking (ONNX). Trả về true nếu thành công. */
    virtual bool loadTrackingModel(const QString& modelPath) = 0;

    // --- State Query ---
    /** Kiểm tra detection model đã sẵn sàng chưa. */
    virtual bool isDetectionReady() const = 0;

    /** Kiểm tra segmentation model đã sẵn sàng chưa. */
    virtual bool isSegmentationReady() const = 0;

    /** Kiểm tra tracking model đã sẵn sàng chưa. */
    virtual bool isTrackingReady() const = 0;

    // --- Inference ---
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
