#include "AIPredictionLogger.h"

#include "AppConfig.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QThreadPool>
#include <opencv2/imgcodecs.hpp>

namespace AIPredictionLogger {

void logAsync(const QString &mode, const QString &sourceImagePath, const cv::Mat &resultImage)
{
    const QString currentDate = QDateTime::currentDateTime().toString("yyyy-MM-dd");
    const QString predictDir = AppConfig::instance().predictDir(mode) + "/" + currentDate;
    QDir().mkpath(predictDir);

    const QString logPath = predictDir + "/" + QFileInfo(sourceImagePath).baseName() + "_" +
                            QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";

    const cv::Mat resultClone = resultImage.clone();
    QThreadPool::globalInstance()->start([logPath, resultClone]() {
        cv::imwrite(logPath.toStdString(), resultClone);
    });
}

}
