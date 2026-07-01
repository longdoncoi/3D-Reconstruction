#include "AIService.h"

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

cv::Mat AIService::runDetection(const cv::Mat& inputImage) {
    return m_processor->runObjectDetection(inputImage);
}

cv::Mat AIService::runSegmentation(const cv::Mat& inputImage) {
    return m_processor->runSegmentation(inputImage);
}

bool AIService::loadTrackingModel(const QString& modelPath) {
    return m_processor->loadTrackingModel(modelPath);
}

bool AIService::isTrackingReady() const {
    return m_processor->isTrackingModelLoaded();
}

cv::Mat AIService::runTracking(const cv::Mat& inputImage) {
    return m_processor->runTracking(inputImage);
}

void AIService::resetTrackingState() {
    m_processor->resetTrackingState();
}
