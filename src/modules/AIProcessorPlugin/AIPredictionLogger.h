#ifndef AI_PREDICTION_LOGGER_H
#define AI_PREDICTION_LOGGER_H

#include <QString>
#include <opencv2/core.hpp>

namespace AIPredictionLogger {

void logAsync(const QString &mode, const QString &sourceImagePath, const cv::Mat &resultImage);

}

#endif // AI_PREDICTION_LOGGER_H
