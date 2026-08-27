#ifndef AIPROCESSOR_H
#define AIPROCESSOR_H

#include <QString>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "Global.h"

struct AIResult {
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<float> mask_coeffs;
};

class APP_EXPORT AIProcessor {
public:
    AIProcessor();
    ~AIProcessor();

    bool loadDetectionModel(const QString& modelPath);
    bool loadSegmentationModel(const QString& modelPath);
    bool loadTrackingModel(const QString& modelPath);

    bool isDetectionModelLoaded() const { return isDetModelLoaded; }
    bool isSegmentationModelLoaded() const { return isSegModelLoaded; }
    bool isTrackingModelLoaded() const { return isTrackingLoaded; }

    /** Set confidence threshold [0.0, 1.0] used in detection & segmentation. */
    void setConfidenceThreshold(float threshold) { m_confidenceThreshold = threshold; }
    float confidenceThreshold() const { return m_confidenceThreshold; }

    // Returns image with drawn bounding boxes
    cv::Mat runObjectDetection(const cv::Mat& inputImage);

    // Returns image with drawn segmentation masks
    cv::Mat runSegmentation(const cv::Mat& inputImage);

    // Tracking
    cv::Mat runTracking(const cv::Mat& inputImage);
    void resetTrackingState();

private:
    void tryInitGPUProvider();
    Ort::Value prepareInputTensor(const cv::Mat& img, int width, int height, std::vector<float>& tensorValues);

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::unique_ptr<Ort::Session> detSession;
    std::unique_ptr<Ort::Session> segSession;
    std::unique_ptr<Ort::Session> trackingSession;

    bool isDetModelLoaded;
    bool isSegModelLoaded;
    bool isTrackingLoaded;
    float m_confidenceThreshold = 0.5f;

    // Tracking state
    std::mutex m_trackingMutex;
    std::map<int, cv::Rect> currentTracks;
    int nextTrackId = 0;

    // Persists the TRT engine cache path for the lifetime of sessionOptions.
    // trt_options.trt_engine_cache_path points into this buffer.
    QByteArray m_trtCachePathBytes;
};

#endif // AIPROCESSOR_H
