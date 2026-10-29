#include "AIModelLoadService.h"
#include "IAIModelAdmin.h"

#include <QFileInfo>

#include <exception>

AIModelLoadService::AIModelLoadService(IAIModelAdmin* ai, QObject* parent)
    : IAIModelLoadService(parent)
    , m_ai(ai)
{
}

AIModelLoadService::~AIModelLoadService()
{
    m_cancel.store(true);

    // The join is intentionally UNBOUNDED. The worker captures `this` and
    // calls into m_ai (IAIModelAdmin), which AppShell destroys only after
    // this service — so the worker MUST be finished before this destructor
    // returns. A bounded wait would return with the worker still alive and
    // let it touch a freed AIService / emit into a destroyed QObject (a
    // use-after-free crash, not a slow shutdown). ONNX cannot be interrupted
    // mid-call, so the only safe option is to wait it out; the app is exiting
    // anyway.
    m_tasks.waitForAll(); // teardown-unbounded: the worker captures `this` and m_ai; it must finish before this destructor returns
}

void AIModelLoadService::loadModelsAsync(const QString& detectionPath,
                                         const QString& segmentationPath,
                                         const QString& trackingPath)
{
    if (!m_ai || m_running.exchange(true))
        return;

    m_cancel.store(false);

    // The previous task has completed (m_running was false), so its
    // semaphore is released. Reap it before registering the new task: a
    // long-lived service that reloads models must not accumulate one tracked
    // semaphore per load for the whole session.
    m_tasks.reapFinished();

    // Dedicated thread, not the global pool: a GUI caller must never execute
    // the load inline (QThreadPool::doStart does that when it has no worker),
    // and the tracker gives teardown exactly one join point.
    (void)blocking::run(
        [this, detectionPath, segmentationPath, trackingPath] {
            runLoad(detectionPath, segmentationPath, trackingPath);
        },
        &m_tasks);
}

void AIModelLoadService::runLoad(const QString& detectionPath,
                                 const QString& segmentationPath,
                                 const QString& trackingPath)
{
    bool ok = true;
    QString error;

    auto loadOne = [&](const QString& path, const char* stage,
                       bool (IAIModelAdmin::*loader)(const QString&)) {
        if (m_cancel.load())
            return;
        if (path.isEmpty() || !QFileInfo::exists(path))
            return;

        emit stageChanged(QString::fromLatin1(stage));
        if (!(m_ai->*loader)(path)) {
            ok = false;
            if (!error.isEmpty())
                error += QStringLiteral("; ");
            error += QStringLiteral("Failed to load %1 model: %2")
                         .arg(QString::fromLatin1(stage), path);
        }
    };

    // A loader that throws (ONNX Runtime and the wrappers can) must
    // not leave m_running stuck at true and swallow the terminal
    // signal: every later loadModelsAsync would then be rejected and
    // every consumer would believe a load is still in flight forever.
    // Catching here also keeps the exception from reaching
    // std::terminate on the worker thread.
    try {
        loadOne(detectionPath, "detection", &IAIModelAdmin::loadDetectionModel);
        loadOne(segmentationPath, "segmentation", &IAIModelAdmin::loadSegmentationModel);
        loadOne(trackingPath, "tracking", &IAIModelAdmin::loadTrackingModel);
    } catch (const std::exception& e) {
        ok = false;
        const QString detail = QStringLiteral("Model load failed: %1")
                                   .arg(QString::fromUtf8(e.what()));
        error = error.isEmpty() ? detail : error + QStringLiteral("; ") + detail;
    } catch (...) {
        ok = false;
        const QString detail = QStringLiteral("Model load failed with an unknown error.");
        error = error.isEmpty() ? detail : error + QStringLiteral("; ") + detail;
    }

    m_running.store(false);
    if (m_cancel.load())
        emit cancelled();
    else
        emit finished(ok, error);
}

void AIModelLoadService::cancel()
{
    // ONNX Runtime offers no mid-call cancellation hook, so the load itself
    // cannot be interrupted: the flag is observed between models by runLoad().
    // The worker owns its own thread, so there is no pool runnable to cancel.
    m_cancel.store(true);
}
