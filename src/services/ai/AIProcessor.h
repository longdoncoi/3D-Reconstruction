#ifndef AIPROCESSOR_H
#define AIPROCESSOR_H

#include <QString>
#include <QString>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <mutex>

struct AIResult {
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<float> mask_coeffs;
};

#include "Global.h"

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

    // Returns image with drawn bounding boxes
    cv::Mat runObjectDetection(const cv::Mat& inputImage);
    
    // Returns image with drawn segmentation masks
    cv::Mat runSegmentation(const cv::Mat& inputImage);

    // Tracking
    cv::Mat runTracking(const cv::Mat& inputImage);
    void resetTrackingState();

private:
    void applyNMS(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, 
                  float scoreThreshold, float nmsThreshold, std::vector<int>& indices);
    
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

    // Tracking state
    std::mutex m_trackingMutex;
    std::map<int, cv::Rect> currentTracks;
    int nextTrackId = 0;

    // Persists the TRT engine cache path for the lifetime of sessionOptions.
    // trt_options.trt_engine_cache_path points into this buffer.
    QByteArray m_trtCachePathBytes;
};

#endif // AIPROCESSOR_H
