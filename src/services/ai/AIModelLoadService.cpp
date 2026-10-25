#include "AIModelLoadService.h"
#include "IAIModelAdmin.h"

#include <QFileInfo>
#include <QtConcurrent>

#include <exception>

AIModelLoadService::AIModelLoadService(IAIModelAdmin* ai, QObject* parent)
    : IAIModelLoadService(parent)
    , m_ai(ai)
{
    m_pool.setMaxThreadCount(1);
}

AIModelLoadService::~AIModelLoadService()
{
    m_cancel.store(true);
    if (m_future.isRunning())
        m_future.cancel();

    // The join is intentionally UNBOUNDED. The worker captures `this` and
    // calls into m_ai (IAIModelAdmin), which AppShell destroys only after
    // this service — so the worker MUST be finished before this destructor
    // returns. A bounded wait would return with the worker still alive and
    // let it touch a freed AIService / emit into a destroyed QObject (a
    // use-after-free crash, not a slow shutdown). ONNX cannot be interrupted
    // mid-call, so the only safe option is to wait it out; the app is exiting
    // anyway.
    m_pool.waitForDone(); // teardown-unbounded: the worker captures `this` and m_ai; it must finish before this destructor returns
}

void AIModelLoadService::loadModelsAsync(const QString& detectionPath,
                                         const QString& segmentationPath,
                                         const QString& trackingPath)
{
    if (!m_ai || m_running.exchange(true))
        return;

    m_cancel.store(false);

    m_future = QtConcurrent::run(
        &m_pool,
        [this, detectionPath, segmentationPath, trackingPath]() {
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
            // std::terminate on the pool thread.
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
        });
}

void AIModelLoadService::cancel()
{
    m_cancel.store(true);
    if (m_future.isRunning())
        m_future.cancel();
}
