#include "AIYoloPostProcessor.h"
#include <QtTest>
#include <algorithm>

class TestAIYoloPostProcessor : public QObject {
  Q_OBJECT

private slots:
  void testModelInputSizeDefault() {
    std::vector<int64_t> shape{-1, 3, 0, 0};
    cv::Size sz = AIYoloPostProcessor::modelInputSize(shape);
    QCOMPARE(sz.width, AIYoloPostProcessor::kDefaultModelInputSize);
    QCOMPARE(sz.height, AIYoloPostProcessor::kDefaultModelInputSize);
  }

  void testModelInputSizeDynamic() {
    std::vector<int64_t> shape{-1, 3, 640, 640};
    cv::Size sz = AIYoloPostProcessor::modelInputSize(shape);
    QCOMPARE(sz.width, 640);
    QCOMPARE(sz.height, 640);
  }

  void testModelInputSizeDynamicDifferent() {
    std::vector<int64_t> shape{1, 3, 32, 32};
    cv::Size sz = AIYoloPostProcessor::modelInputSize(shape);
    QCOMPARE(sz.width, 32);
    QCOMPARE(sz.height, 32);
  }

  void testParsePredictionsFiltersByScore() {
    // Shape: rows=2, cols=5 (x,y,w,h + 1 class)
    cv::Mat preds(2, 5, CV_32F);
    // Row 0: high confidence
    preds.at<float>(0, 0) = 10.0f; // cx
    preds.at<float>(0, 1) = 10.0f; // cy
    preds.at<float>(0, 2) = 10.0f; // w
    preds.at<float>(0, 3) = 10.0f; // h
    preds.at<float>(0, 4) = 0.8f;  // cls0
    // Row 1: low confidence
    preds.at<float>(1, 0) = 5.0f;
    preds.at<float>(1, 1) = 5.0f;
    preds.at<float>(1, 2) = 5.0f;
    preds.at<float>(1, 3) = 5.0f;
    preds.at<float>(1, 4) = 0.1f;

    YoloDetections det = AIYoloPostProcessor::parsePredictions(
        preds, cv::Size(100, 100), cv::Size(20, 20), 0.5f);
    QCOMPARE(det.boxes.size(), 1);
    QCOMPARE(det.confidences.size(), 1);
    QCOMPARE(det.classIds.size(), 1);
    QVERIFY(det.confidences[0] >= 0.5f);
  }

  void testParsePredictionsWithMasks() {
    // cols = 4 + numClasses + maskCoeffs
    const int numClasses = 1;
    const int maskCoeffs = 2;
    cv::Mat preds(1, 4 + numClasses + maskCoeffs, CV_32F);
    preds.at<float>(0, 0) = 0.0f;
    preds.at<float>(0, 1) = 0.0f;
    preds.at<float>(0, 2) = 2.0f;
    preds.at<float>(0, 3) = 2.0f;
    preds.at<float>(0, 4) = 0.9f;
    preds.at<float>(0, 5) = 0.25f;
    preds.at<float>(0, 6) = 0.75f;

    YoloDetections det = AIYoloPostProcessor::parsePredictions(
        preds, cv::Size(64, 64), cv::Size(64, 64), 0.5f, maskCoeffs);
    QCOMPARE(det.boxes.size(), 1);
    QCOMPARE((int)det.maskCoefficients.size(), 1);
    QCOMPARE((int)det.maskCoefficients[0].size(), maskCoeffs);
  }

  void testApplyNmsKeepsBest() {
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    boxes.push_back(cv::Rect(0, 0, 10, 10));
    confs.push_back(0.9f);
    boxes.push_back(cv::Rect(1, 1, 10, 10));
    confs.push_back(0.8f);
    std::vector<int> keep = AIYoloPostProcessor::applyNms(boxes, confs, 0.5f);
    QCOMPARE((int)keep.size(), 1);
    QCOMPARE(keep[0], 0);
  }

  void testApplyNmsNoOverlap() {
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    boxes.push_back(cv::Rect(0, 0, 5, 5));
    confs.push_back(0.9f);
    boxes.push_back(cv::Rect(10, 10, 5, 5));
    confs.push_back(0.8f);
    std::vector<int> keep = AIYoloPostProcessor::applyNms(boxes, confs, 0.5f);
    QCOMPARE((int)keep.size(), 2);
  }

  void testApplyNmsEmpty() {
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> keep = AIYoloPostProcessor::applyNms(boxes, confs, 0.5f);
    QVERIFY(keep.empty());
  }
};

QTEST_GUILESS_MAIN(TestAIYoloPostProcessor)
#include "TestAIYoloPostProcessor.moc"
