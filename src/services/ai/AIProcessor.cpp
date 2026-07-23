#include "AIProcessor.h"
#include <QDebug>
#include <QDir>
#include <algorithm>
#include <numeric>
#include <opencv2/imgproc.hpp>

AIProcessor::AIProcessor() : isDetModelLoaded(false), isSegModelLoaded(false), isTrackingLoaded(false) {
  env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "AIProcessor");
  sessionOptions = std::make_unique<Ort::SessionOptions>();
  sessionOptions->SetIntraOpNumThreads(1);
  sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  tryInitGPUProvider();
}

void AIProcessor::tryInitGPUProvider() {
  try {
    auto providers = Ort::GetAvailableProviders();
    bool trt_available = false;
    bool cuda_available = false;

    for (const auto &p : providers) {
      qDebug() << "ONNX Runtime available provider:" << p.c_str();
      if (p == "TensorrtExecutionProvider") trt_available = true;
      else if (p == "CUDAExecutionProvider") cuda_available = true;
    }

    // TensorRT builds an engine for every model at startup. On systems with
    // limited VRAM or page-file space this can exhaust memory before CUDA or
    // the AI assistant has a chance to start. CUDA remains GPU-accelerated
    // and is the safe default; TensorRT is an explicit opt-in.
    const bool enableTensorRT =
        qEnvironmentVariableIntValue("RECONSTRUCTION_ENABLE_TENSORRT") == 1;

    if (trt_available && enableTensorRT) {
        // Use absolute path to avoid issues with relative paths in TRT cache
        QString cacheDir = QDir::currentPath() + "/models/cache";
        QDir().mkpath(cacheDir);
        // Store in class member so the pointer outlives this function —
        // trt_options.trt_engine_cache_path will reference this buffer.
        m_trtCachePathBytes = cacheDir.toLocal8Bit();

        // CRITICAL: Zero-initialize the struct! Without this, unset const char*
        // fields contain garbage values causing read access violations inside TRT.
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = 0;
        trt_options.trt_max_partition_iterations = 100;
        trt_options.trt_min_subgraph_size = 5;
        trt_options.trt_max_workspace_size = 256ULL << 20; // 256 MB
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = m_trtCachePathBytes.constData();
        trt_options.trt_fp16_enable = 1; // Enable FP16 for better performance
        sessionOptions->AppendExecutionProvider_TensorRT(trt_options);
        qDebug() << "ONNX Runtime: Using TensorRT Execution Provider (High Performance)";
        qDebug() << "TensorRT cache path:" << cacheDir;
    } else if (cuda_available) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        sessionOptions->AppendExecutionProvider_CUDA(cuda_options);
        if (trt_available) {
            qDebug() << "ONNX Runtime: TensorRT is available but disabled by default "
                        "to avoid startup memory exhaustion; using CUDA. "
                        "Set RECONSTRUCTION_ENABLE_TENSORRT=1 to opt in.";
        } else {
            qDebug() << "ONNX Runtime: Using CUDA Execution Provider (GPU)";
        }
    } else {
        qDebug() << "ONNX Runtime: GPU providers not found, using CPU fallback";
    }
  } catch (const std::exception &e) {
    qWarning() << "Failed to initialize GPU provider, falling back to CPU:" << e.what();
  }
}

AIProcessor::~AIProcessor() {}

bool AIProcessor::loadDetectionModel(const QString &modelPath) {
  try {
    std::wstring w_modelPath = modelPath.toStdWString();
    detSession = std::make_unique<Ort::Session>(*env, w_modelPath.c_str(),
                                                *sessionOptions);
    isDetModelLoaded = true;
    qDebug() << "Detection model loaded successfully with ORT:" << modelPath;
    return true;
  } catch (const Ort::Exception &e) {
    qWarning() << "ORT Exception during loading detection model:" << e.what();
    isDetModelLoaded = false;
    return false;
  }
}

bool AIProcessor::loadSegmentationModel(const QString &modelPath) {
  try {
    std::wstring w_modelPath = modelPath.toStdWString();
    segSession = std::make_unique<Ort::Session>(*env, w_modelPath.c_str(),
                                                *sessionOptions);
    isSegModelLoaded = true;
    qDebug() << "Segmentation model loaded successfully with ORT:" << modelPath;
    return true;
  } catch (const Ort::Exception &e) {
    qWarning() << "ORT Exception during loading segmentation model:"
               << e.what();
    isSegModelLoaded = false;
    return false;
  }
}

bool AIProcessor::loadTrackingModel(const QString &modelPath) {
  try {
    std::wstring w_modelPath = modelPath.toStdWString();
    trackingSession = std::make_unique<Ort::Session>(*env, w_modelPath.c_str(),
                                                *sessionOptions);
    isTrackingLoaded = true;
    qDebug() << "Tracking model loaded successfully with ORT:" << modelPath;
    return true;
  } catch (const Ort::Exception &e) {
    qWarning() << "ORT Exception during loading tracking model:" << e.what();
    isTrackingLoaded = false;
    return false;
  }
}

cv::Mat AIProcessor::runObjectDetection(const cv::Mat &inputImage) {
  if (!isDetModelLoaded || inputImage.empty()) {
    qWarning() << "Detection model not loaded or input image is empty.";
    return inputImage;
  }

  const float SCORE_THRESHOLD = 0.5f;
  const float NMS_THRESHOLD = 0.45f;
  const int INPUT_WIDTH = 640;
  const int INPUT_HEIGHT = 640;
  const int CHANNELS = 3;

  std::vector<float> inputTensorValues;
  Ort::Value inputTensor = prepareInputTensor(inputImage, INPUT_WIDTH, INPUT_HEIGHT, inputTensorValues);

  Ort::AllocatorWithDefaultOptions allocator;
  auto inputNamePtr = detSession->GetInputNameAllocated(0, allocator);
  const char *inputNames[] = {inputNamePtr.get()};

  auto outputNamePtr = detSession->GetOutputNameAllocated(0, allocator);
  const char *outputNames[] = {outputNamePtr.get()};

  auto outputTensors = detSession->Run(Ort::RunOptions{nullptr}, inputNames,
                                       &inputTensor, 1, outputNames, 1);

  float *outputData = outputTensors[0].GetTensorMutableData<float>();
  auto dims = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  cv::Mat outMat(dims[1], dims[2], CV_32F, outputData);
  outMat = outMat.t();

  int num_preds = outMat.rows;
  int num_classes = outMat.cols - 4;

  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  float x_factor = inputImage.cols / (float)INPUT_WIDTH;
  float y_factor = inputImage.rows / (float)INPUT_HEIGHT;

  for (int i = 0; i < num_preds; ++i) {
    float *data = outMat.ptr<float>(i);

    float max_score = -1.f;
    int class_id = -1;
    for (int j = 0; j < num_classes; ++j) {
      if (data[4 + j] > max_score) {
        max_score = data[4 + j];
        class_id = j;
      }
    }

    if (max_score >= SCORE_THRESHOLD) {
      float cx = data[0];
      float cy = data[1];
      float w = data[2];
      float h = data[3];

      int left = int((cx - 0.5f * w) * x_factor);
      int top = int((cy - 0.5f * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);

      classIds.push_back(class_id);
      confidences.push_back(max_score);
      boxes.push_back(cv::Rect(left, top, width, height));
    }
  }

  std::vector<int> indices;
  applyNMS(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

  cv::Mat resultImage = inputImage.clone();
  for (int idx : indices) {
    cv::Rect box = boxes[idx];
    cv::rectangle(resultImage, box, cv::Scalar(0, 255, 0), 2);

    QString label = QString("ID %1: %2%")
                        .arg(classIds[idx])
                        .arg(int(confidences[idx] * 100));
    int baseLine;
    cv::Size labelSize = cv::getTextSize(
        label.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(resultImage,
                  cv::Point(box.x, box.y - labelSize.height - baseLine),
                  cv::Point(box.x + labelSize.width, box.y),
                  cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(resultImage, label.toStdString(),
                cv::Point(box.x, box.y - baseLine), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 0, 0), 1);
  }

  return resultImage;
}

cv::Mat AIProcessor::runSegmentation(const cv::Mat &inputImage) {
  if (!isSegModelLoaded || inputImage.empty()) {
    qWarning() << "Segmentation model not loaded or input image is empty.";
    return inputImage;
  }

  const float SCORE_THRESHOLD = 0.5f;
  const float NMS_THRESHOLD = 0.45f;
  const float MASK_THRESHOLD = 0.5f;
  const int INPUT_WIDTH = 640;
  const int INPUT_HEIGHT = 640;
  const int CHANNELS = 3;

  std::vector<float> inputTensorValues;
  Ort::Value inputTensor = prepareInputTensor(inputImage, INPUT_WIDTH, INPUT_HEIGHT, inputTensorValues);

  Ort::AllocatorWithDefaultOptions allocator;
  auto inputNamePtr = segSession->GetInputNameAllocated(0, allocator);
  const char *inputNames[] = {inputNamePtr.get()};

  auto outNamePtr0 = segSession->GetOutputNameAllocated(0, allocator);
  auto outNamePtr1 = segSession->GetOutputNameAllocated(1, allocator);
  const char *outputNames[] = {outNamePtr0.get(), outNamePtr1.get()};

  auto outputTensors = segSession->Run(Ort::RunOptions{nullptr}, inputNames,
                                       &inputTensor, 1, outputNames, 2);

  float *outData0 = outputTensors[0].GetTensorMutableData<float>();
  auto dims0 = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  float *outData1 = outputTensors[1].GetTensorMutableData<float>();
  auto dims1 = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

  cv::Mat boxes_mat, proto_mat;
  if (dims0.size() == 4) {
    int sizes0[] = {(int)dims0[0], (int)dims0[1], (int)dims0[2], (int)dims0[3]};
    proto_mat = cv::Mat(4, sizes0, CV_32F, outData0);
    boxes_mat = cv::Mat(dims1[1], dims1[2], CV_32F, outData1);
  } else {
    int sizes1[] = {(int)dims1[0], (int)dims1[1], (int)dims1[2], (int)dims1[3]};
    proto_mat = cv::Mat(4, sizes1, CV_32F, outData1);
    boxes_mat = cv::Mat(dims0[1], dims0[2], CV_32F, outData0);
  }

  boxes_mat = boxes_mat.t();

  int num_preds = boxes_mat.rows;
  int mask_coeffs_count = proto_mat.size[1];
  int num_classes = boxes_mat.cols - 4 - mask_coeffs_count;

  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> masks_coeffs_list;

  float x_factor = inputImage.cols / (float)INPUT_WIDTH;
  float y_factor = inputImage.rows / (float)INPUT_HEIGHT;

  for (int i = 0; i < num_preds; ++i) {
    float *data = boxes_mat.ptr<float>(i);

    float max_score = -1.f;
    int class_id = -1;
    for (int j = 0; j < num_classes; ++j) {
      if (data[4 + j] > max_score) {
        max_score = data[4 + j];
        class_id = j;
      }
    }

    if (max_score >= SCORE_THRESHOLD) {
      float cx = data[0];
      float cy = data[1];
      float w = data[2];
      float h = data[3];

      int left = int((cx - 0.5f * w) * x_factor);
      int top = int((cy - 0.5f * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);

      classIds.push_back(class_id);
      confidences.push_back(max_score);
      boxes.push_back(cv::Rect(left, top, width, height));

      std::vector<float> coeffs(data + 4 + num_classes,
                                data + 4 + num_classes + mask_coeffs_count);
      masks_coeffs_list.push_back(coeffs);
    }
  }

  std::vector<int> indices;
  applyNMS(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

  cv::Mat resultImage = inputImage.clone();
  cv::Mat color_mask = cv::Mat::zeros(resultImage.size(), CV_8UC3);

  int proto_h = proto_mat.size[2];
  int proto_w = proto_mat.size[3];
  cv::Mat proto(mask_coeffs_count, proto_h * proto_w, CV_32F,
                proto_mat.ptr<float>());

  for (int idx : indices) {
    cv::Rect box = boxes[idx];

    cv::Mat coeffs_mat(1, mask_coeffs_count, CV_32F,
                       masks_coeffs_list[idx].data());
    cv::Mat mask_mat = coeffs_mat * proto;
    mask_mat = mask_mat.reshape(1, proto_h);

    cv::exp(-mask_mat, mask_mat);
    mask_mat = 1.0f / (1.0f + mask_mat);

    cv::resize(mask_mat, mask_mat, resultImage.size());
    cv::Mat binary_mask = mask_mat > MASK_THRESHOLD;

    cv::Rect img_rect(0, 0, resultImage.cols, resultImage.rows);
    cv::Rect valid_box = box & img_rect;

    if (valid_box.area() > 0) {
      cv::Mat cropped_mask = cv::Mat::zeros(binary_mask.size(), CV_8U);
      binary_mask(valid_box).copyTo(cropped_mask(valid_box));

      // Fill red mask
      color_mask.setTo(cv::Scalar(0, 0, 255), cropped_mask);

      // Draw green contour around the object
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(cropped_mask, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(resultImage, contours, -1, cv::Scalar(0, 255, 0), 2);
    }

    // cv::rectangle(resultImage, box, cv::Scalar(0, 0, 255), 2);
  }

  cv::addWeighted(resultImage, 1.0, color_mask, 0.5, 0, resultImage);

  return resultImage;
}

void AIProcessor::resetTrackingState() {
    std::lock_guard<std::mutex> lock(m_trackingMutex);
    currentTracks.clear();
    nextTrackId = 0;
}

cv::Mat AIProcessor::runTracking(const cv::Mat &inputImage) {
  if (!isTrackingLoaded || inputImage.empty()) {
    qWarning() << "Tracking model not loaded or input image is empty.";
    return inputImage;
  }

  const float SCORE_THRESHOLD = 0.5f;
  const float NMS_THRESHOLD = 0.45f;
  const int INPUT_WIDTH = 640;
  const int INPUT_HEIGHT = 640;
  const int CHANNELS = 3;

  std::vector<float> inputTensorValues;
  Ort::Value inputTensor = prepareInputTensor(inputImage, INPUT_WIDTH, INPUT_HEIGHT, inputTensorValues);

  Ort::AllocatorWithDefaultOptions allocator;
  auto inputNamePtr = trackingSession->GetInputNameAllocated(0, allocator);
  const char *inputNames[] = {inputNamePtr.get()};

  auto outputNamePtr = trackingSession->GetOutputNameAllocated(0, allocator);
  const char *outputNames[] = {outputNamePtr.get()};

  auto outputTensors = trackingSession->Run(Ort::RunOptions{nullptr}, inputNames,
                                       &inputTensor, 1, outputNames, 1);

  float *outputData = outputTensors[0].GetTensorMutableData<float>();
  auto dims = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

  cv::Mat outMat(dims[1], dims[2], CV_32F, outputData);
  outMat = outMat.t();

  int num_preds = outMat.rows;
  int num_classes = outMat.cols - 4;

  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  float x_factor = inputImage.cols / (float)INPUT_WIDTH;
  float y_factor = inputImage.rows / (float)INPUT_HEIGHT;

  for (int i = 0; i < num_preds; ++i) {
    float *data = outMat.ptr<float>(i);

    float max_score = -1.f;
    int class_id = -1;
    for (int j = 0; j < num_classes; ++j) {
      if (data[4 + j] > max_score) {
        max_score = data[4 + j];
        class_id = j;
      }
    }

    if (max_score >= SCORE_THRESHOLD) {
      float cx = data[0];
      float cy = data[1];
      float w = data[2];
      float h = data[3];

      int left = int((cx - 0.5f * w) * x_factor);
      int top = int((cy - 0.5f * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);

      classIds.push_back(class_id);
      confidences.push_back(max_score);
      boxes.push_back(cv::Rect(left, top, width, height));
    }
  }

  std::vector<int> indices;
  applyNMS(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

  std::vector<cv::Rect> current_frame_boxes;
  for (int idx : indices) {
      current_frame_boxes.push_back(boxes[idx]);
  }

  std::lock_guard<std::mutex> lock(m_trackingMutex);

  // Simple IOU Tracking
  std::map<int, cv::Rect> newTracks;
  std::vector<bool> matched_current(current_frame_boxes.size(), false);
  const float IOU_THRESHOLD = 0.3f;

  for (auto const& [track_id, prev_box] : currentTracks) {
      int best_match = -1;
      float best_iou = 0.0f;
      for (size_t i = 0; i < current_frame_boxes.size(); ++i) {
          if (matched_current[i]) continue;
          
          cv::Rect curr_box = current_frame_boxes[i];
          float inter_area = (prev_box & curr_box).area();
          float union_area = prev_box.area() + curr_box.area() - inter_area;
          float iou = inter_area / union_area;
          
          if (iou > best_iou && iou > IOU_THRESHOLD) {
              best_iou = iou;
              best_match = i;
          }
      }
      if (best_match != -1) {
          matched_current[best_match] = true;
          newTracks[track_id] = current_frame_boxes[best_match];
      }
  }

  // Add unmatched to new tracks
  for (size_t i = 0; i < current_frame_boxes.size(); ++i) {
      if (!matched_current[i]) {
          newTracks[nextTrackId++] = current_frame_boxes[i];
      }
  }

  currentTracks = newTracks;

  cv::Mat resultImage = inputImage.clone();
  for (auto const& [track_id, box] : currentTracks) {
    // Generate color based on track_id
    cv::Scalar color((track_id * 37) % 255, (track_id * 73) % 255, (track_id * 149) % 255);
    cv::rectangle(resultImage, box, color, 2);

    QString label = QString("ID: %1").arg(track_id);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(
        label.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(resultImage,
                  cv::Point(box.x, box.y - labelSize.height - baseLine),
                  cv::Point(box.x + labelSize.width, box.y),
                  color, cv::FILLED);
    cv::putText(resultImage, label.toStdString(),
                cv::Point(box.x, box.y - baseLine), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 255), 1);
  }

  return resultImage;
}

Ort::Value AIProcessor::prepareInputTensor(const cv::Mat& img, int width, int height, std::vector<float>& tensorValues) {
  cv::Mat resizedImage;
  cv::resize(img, resizedImage, cv::Size(width, height));
  cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

  cv::Mat floatImage;
  resizedImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  cv::split(floatImage, channels);

  tensorValues.clear();
  tensorValues.reserve(3 * width * height);
  for (int i = 0; i < 3; ++i) {
    tensorValues.insert(tensorValues.end(), (float*)channels[i].data,
                        (float*)channels[i].data + width * height);
  }

  std::vector<int64_t> inputDims = {1, 3, (int64_t)height, (int64_t)width};
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  
  return Ort::Value::CreateTensor<float>(
      memoryInfo, tensorValues.data(), tensorValues.size(),
      inputDims.data(), inputDims.size());
}

void AIProcessor::applyNMS(const std::vector<cv::Rect> &boxes,
                           const std::vector<float> &confidences,
                           float scoreThreshold, float nmsThreshold,
                           std::vector<int> &indices) {
  indices.clear();
  if (boxes.empty())
    return;

  std::vector<int> sorted_indices(boxes.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int i, int j) { return confidences[i] > confidences[j]; });

  std::vector<bool> is_suppressed(boxes.size(), false);

  for (size_t i = 0; i < sorted_indices.size(); ++i) {
    int idx1 = sorted_indices[i];
    if (is_suppressed[idx1])
      continue;

    indices.push_back(idx1);

    for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
      int idx2 = sorted_indices[j];
      if (is_suppressed[idx2])
        continue;

      cv::Rect inter = boxes[idx1] & boxes[idx2];
      float inter_area = (float)inter.area();
      float union_area =
          (float)(boxes[idx1].area() + boxes[idx2].area()) - inter_area;
      float iou = inter_area / union_area;

      if (iou > nmsThreshold) {
        is_suppressed[idx2] = true;
      }
    }
  }
}
