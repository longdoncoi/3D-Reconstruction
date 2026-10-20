#include "AIModelLoadService.h"
#include "IAIService.h"

#include <QFileInfo>
#include <QtConcurrent>

AIModelLoadService::AIModelLoadService(IAIService* ai, QObject* parent)
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

    // The task is compiled into AppCore, not into a plugin DLL, so waiting
    // here is safe from an unloading standpoint. AppShell destroys this
    // service before IAIService, keeping m_ai alive for the worker.
    m_pool.waitForDone();
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
                               bool (IAIService::*loader)(const QString&)) {
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

            loadOne(detectionPath, "detection", &IAIService::loadDetectionModel);
            loadOne(segmentationPath, "segmentation", &IAIService::loadSegmentationModel);
            loadOne(trackingPath, "tracking", &IAIService::loadTrackingModel);

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
