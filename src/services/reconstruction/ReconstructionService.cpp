#include "ReconstructionService.h"
#include <QDir>

ReconstructionService::ReconstructionService()
    : m_pipeline(std::make_unique<ReconstructionPipeline>())
{
}

void ReconstructionService::setImages(const QStringList& paths) {
    if (m_running) return;
    std::vector<QString> stdPaths;
    stdPaths.reserve(static_cast<size_t>(paths.size()));
    for (const QString& p : paths)
        stdPaths.push_back(QDir::toNativeSeparators(p));
    
    std::unique_lock lock(m_mutex);
    m_pipeline->setImages(stdPaths);
}

bool ReconstructionService::loadCameraParams(const QString& filePath) {
    if (m_running) return false;
    std::unique_lock lock(m_mutex);
    return m_pipeline->loadCameraParams(filePath);
}

bool ReconstructionService::reconstruct() {
    if (m_running.exchange(true)) return false; // Already running
    
    bool ok;
    {
        std::unique_lock lock(m_mutex);
        ok = m_pipeline->reconstruct();
    }
    m_running = false;
    return ok;
}

bool ReconstructionService::isRunning() const {
    return m_running;
}

void ReconstructionService::stopReconstruction() {
    // ReconstructionPipeline doesn't have a stop API yet.
    // This is a placeholder — thread termination handled by ReconstructThread.
    m_running = false;
}

QStringList ReconstructionService::getImageList() const {
    std::shared_lock lock(m_mutex);
    QStringList result;
    for (const QString& p : m_pipeline->getImages())
        result.append(p);
    return result;
}

std::vector<cv::Point3f> ReconstructionService::getPointCloud() const {
    if (m_running) return {};
    std::shared_lock lock(m_mutex);
    return m_pipeline->getPointCloud();
}

std::vector<cv::Vec3b> ReconstructionService::getPointColors() const {
    if (m_running) return {};
    std::shared_lock lock(m_mutex);
    return m_pipeline->getPointColors();
}

bool ReconstructionService::hasResult() const {
    if (m_running) return false;
    std::shared_lock lock(m_mutex);
    return !m_pipeline->getPointCloud().empty();
}

void ReconstructionService::processPointCloud() {
    if (m_running) return;
    std::unique_lock lock(m_mutex);
    m_pipeline->processPointCloud();
}
