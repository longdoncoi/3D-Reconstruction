#ifndef IAI_MODELLOADSERVICE_H
#define IAI_MODELLOADSERVICE_H

#include <QObject>
#include <QString>
#include "Global.h"

/**
 * @brief Application-lifetime owner of asynchronous AI model loading.
 *
 * Model loading is a long-running operation that must not live in a plugin
 * DLL: a plugin can be unloaded while an AppCore-owned worker is still
 * executing. This service is owned by AppShell (AppCore), so cancellation and
 * teardown are bounded by the application lifetime instead of the plugin
 * lifetime. The plugin only observes progress through the signals below.
 *
 * All paths are optional: an empty path or a path that does not exist is
 * skipped. A load already in progress is not restarted.
 */
class APP_EXPORT IAIModelLoadService : public QObject {
    Q_OBJECT
protected:
    explicit IAIModelLoadService(QObject* parent = nullptr) : QObject(parent) {}

public:
    /// ServiceRegistry lookup key (see ServiceRegistry.h).
    static constexpr const char *ServiceKey =
        "com.3dreconstruction.service.IAIModelLoadService";

    virtual ~IAIModelLoadService() = default;

    virtual void loadModelsAsync(const QString& detectionPath,
                                 const QString& segmentationPath,
                                 const QString& trackingPath) = 0;

    /// Cooperative cancellation: no new model is started after this call.
    /// A model currently being loaded by ONNX Runtime cannot be interrupted
    /// mid-call, but the service remains alive until the worker finishes.
    virtual void cancel() = 0;

    virtual bool isLoading() const = 0;

signals:
    /// stageKey is one of "detection", "segmentation", "tracking".
    void stageChanged(const QString& stageKey);

    void finished(bool success, const QString& error);

    void cancelled();
};

#endif // IAI_MODELLOADSERVICE_H
