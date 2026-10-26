#include "AIService.h"

#include "AIInferenceGuard.h"

#include <QDebug>

AIService::AIService(QObject* parent)
    : QObject(parent)
    , m_processor(std::make_unique<AIProcessor>())
{
}

bool AIService::loadDetectionModel(const QString& modelPath) {
    return m_processor->loadDetectionModel(modelPath);
}

bool AIService::loadSegmentationModel(const QString& modelPath) {
    return m_processor->loadSegmentationModel(modelPath);
}

bool AIService::isDetectionReady() const {
    return m_processor->isDetectionModelLoaded();
}

bool AIService::isSegmentationReady() const {
    return m_processor->isSegmentationModelLoaded();
}

AIDetectionOutcome AIService::detectObjects(const cv::Mat& inputImage) {
    // A model that was never loaded cannot infer. The processor would return
    // an EMPTY list, which reads as "detected nothing" — the outcome flag is
    // what tells the caller the backend never ran.
    if (!m_processor->isDetectionModelLoaded()) {
        qWarning() << "[AI] detection requested before the model was loaded";
        return {};
    }
    const ai::DetectionOutcome outcome = ai::runDetectionGuarded(
        "detection",
        [this](const cv::Mat& in) { return m_processor->detectObjects(in); },
        inputImage);
    return AIDetectionOutcome{outcome.ok, outcome.results};
}

cv::Mat AIService::runDetection(const cv::Mat& inputImage) {
    // A model that was never loaded (or failed to load) cannot infer. The
    // processor would return the INPUT unchanged, which reads as "detected
    // nothing" instead of "there is no model": the service reports the truth
    // and the plugin surfaces aiproc.inference_failed.
    if (!m_processor->isDetectionModelLoaded()) {
        qWarning() << "[AI] detection requested before the model was loaded";
        return {};
    }
    return ai::imageOrEmpty(ai::runInferenceGuarded(
        "detection",
        [this](const cv::Mat& in) { return m_processor->runObjectDetection(in); },
        inputImage));
}

cv::Mat AIService::runSegmentation(const cv::Mat& inputImage) {
    if (!m_processor->isSegmentationModelLoaded()) {
        qWarning() << "[AI] segmentation requested before the model was loaded";
        return {};
    }
    return ai::imageOrEmpty(ai::runInferenceGuarded(
        "segmentation",
        [this](const cv::Mat& in) { return m_processor->runSegmentation(in); },
        inputImage));
}

bool AIService::loadTrackingModel(const QString& modelPath) {
    return m_processor->loadTrackingModel(modelPath);
}

bool AIService::isTrackingReady() const {
    return m_processor->isTrackingModelLoaded();
}

cv::Mat AIService::runTracking(const cv::Mat& inputImage) {
    // The tracking path runs on a BACKGROUND thread (VideoTrackerThread):
    // an exception that escaped it would reach std::terminate() and abort
    // the application, so it is contained here like every other inference.
    if (!m_processor->isTrackingModelLoaded()) {
        qWarning() << "[AI] tracking requested before the model was loaded";
        return {};
    }
    return ai::imageOrEmpty(ai::runInferenceGuarded(
        "tracking",
        [this](const cv::Mat& in) { return m_processor->runTracking(in); },
        inputImage));
}

void AIService::resetTrackingState() {
    m_processor->resetTrackingState();
}

void AIService::setConfidenceThreshold(float threshold) {
    m_processor->setConfidenceThreshold(threshold);
}

float AIService::confidenceThreshold() const {
    return m_processor->confidenceThreshold();
}
