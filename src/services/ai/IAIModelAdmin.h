#ifndef IAIMODELADMIN_H
#define IAIMODELADMIN_H

#include <QString>

/**
 * @brief Internal model-management seam between AIModelLoadService and
 *        AIService.
 *
 * NOT plugin-facing. The loaders are synchronous and blocking (ONNX session
 * creation); a plugin must trigger a load through `IAIModelLoadService`
 * (AppCore-owned worker) and observe readiness through `IAIService`. Keeping
 * the loaders off `IAIService` is what makes that the only expressible path:
 * a plugin cannot accidentally freeze the GUI thread with a direct call.
 *
 * Lives in src/services/ai, which is PRIVATE on AppCore, so the type is not
 * part of the plugin ABI.
 */
class IAIModelAdmin {
public:
    virtual ~IAIModelAdmin() = default;

    /** Tải model detection (ONNX). Trả về true nếu thành công. */
    virtual bool loadDetectionModel(const QString& modelPath) = 0;

    /** Tải model segmentation (ONNX). Trả về true nếu thành công. */
    virtual bool loadSegmentationModel(const QString& modelPath) = 0;

    /** Tải model tracking (ONNX). Trả về true nếu thành công. */
    virtual bool loadTrackingModel(const QString& modelPath) = 0;
};

#endif // IAIMODELADMIN_H
