#ifndef AISERVICE_H
#define AISERVICE_H

#include "IAIService.h"
#include "AIProcessor.h"
#include "AIAssistant.h"
#include "Global.h"
#include <memory>
#include <QObject>

/**
 * @brief Concrete implementation của IAIService.
 *
 * Wrap AIProcessor — expose interface sạch cho plugin.
 * Plugin không cần include AIProcessor.h.
 */
class APP_EXPORT AIService : public QObject, public IAIService {
    Q_OBJECT
public:
    explicit AIService(QObject* parent = nullptr);
    ~AIService() override = default;

    // IAIService
    bool loadDetectionModel(const QString& modelPath) override;
    bool loadSegmentationModel(const QString& modelPath) override;
    bool loadTrackingModel(const QString& modelPath) override;
    
    bool isDetectionReady() const override;
    bool isSegmentationReady() const override;
    bool isTrackingReady() const override;
    
    cv::Mat runDetection(const cv::Mat& inputImage) override;
    cv::Mat runSegmentation(const cv::Mat& inputImage) override;
    cv::Mat runTracking(const cv::Mat& inputImage) override;
    void resetTrackingState() override;

    // IAIService methods...
private:
    std::unique_ptr<AIProcessor>  m_processor;
};

#endif // AISERVICE_H
