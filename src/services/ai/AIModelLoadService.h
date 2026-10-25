#ifndef AI_MODELLOADSERVICE_H
#define AI_MODELLOADSERVICE_H

#include "IAIModelLoadService.h"
#include "IAIModelAdmin.h"
#include "Global.h"

#include <QFuture>
#include <QThreadPool>
#include <atomic>

/**
 * @brief AppCore implementation of IAIModelLoadService.
 *
 * Owns its own single-thread QThreadPool so a model load cannot occupy the
 * global pool and so teardown has one explicit join point. The service
 * outlives every plugin (AppShell owns it), which is what makes the async
 * task safe when a plugin is unloaded mid-load.
 */
class APP_EXPORT AIModelLoadService : public IAIModelLoadService {
    Q_OBJECT
public:
    explicit AIModelLoadService(IAIModelAdmin* ai, QObject* parent = nullptr);
    ~AIModelLoadService() override;

    void loadModelsAsync(const QString& detectionPath,
                         const QString& segmentationPath,
                         const QString& trackingPath) override;
    void cancel() override;
    bool isLoading() const override { return m_running.load(); }

private:
    IAIModelAdmin* m_ai = nullptr; ///< not owned; AppShell destroys AI after this service
    QThreadPool m_pool;
    QFuture<void> m_future;
    std::atomic<bool> m_cancel{false};
    std::atomic<bool> m_running{false};
};

#endif // AI_MODELLOADSERVICE_H
