#ifndef AI_MODELLOADSERVICE_H
#define AI_MODELLOADSERVICE_H

#include "IAIModelLoadService.h"
#include "IAIModelAdmin.h"
#include "BlockingCall.h"
#include "Global.h"

#include <atomic>

/**
 * @brief AppCore implementation of IAIModelLoadService.
 *
 * The load runs on a dedicated thread through `blocking::run` — never on the
 * caller's thread. `QtConcurrent::run` looked equivalent but is not: when the
 * pool has no free worker Qt executes the runnable inline, so a model load
 * triggered from a GUI slot could freeze the window for its whole duration.
 * The tracker is the single join point the destructor waits on, and the
 * service outlives every plugin (AppShell owns it), which is what makes the
 * async task safe when a plugin is unloaded mid-load.
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
    /// The worker body. Runs on the dedicated thread, never on the caller's.
    void runLoad(const QString& detectionPath,
                 const QString& segmentationPath,
                 const QString& trackingPath);

    IAIModelAdmin* m_ai = nullptr; ///< not owned; AppShell destroys AI after this service
    blocking::TaskTracker m_tasks;
    std::atomic<bool> m_cancel{false};
    std::atomic<bool> m_running{false};
};

#endif // AI_MODELLOADSERVICE_H
