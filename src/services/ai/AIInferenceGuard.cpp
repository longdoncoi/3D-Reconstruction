#include "AIInferenceGuard.h"

#include <QDebug>

#include <opencv2/core.hpp>

#include <exception>

namespace ai {

namespace {

/// Turns a caught exception into a human-readable message.
QString describeException(const char* stage, const std::exception& e)
{
    return QStringLiteral("%1: %2").arg(QString::fromLatin1(stage),
                                        QString::fromUtf8(e.what()));
}

} // namespace

InferenceOutcome runInferenceGuarded(const char* stage,
                                     const std::function<cv::Mat(const cv::Mat&)>& call,
                                     const cv::Mat& input)
{
    InferenceOutcome outcome;
    try {
        outcome.image = call(input);
        outcome.ok = true;
        return outcome;
    } catch (const cv::Exception& e) {
        outcome.error = describeException(stage, e);
    } catch (const std::exception& e) {
        outcome.error = describeException(stage, e);
    } catch (...) {
        outcome.error = QStringLiteral("%1: failed with an unknown error")
                            .arg(QString::fromLatin1(stage));
    }
    // The call failed: the caller gets the input back rather than an empty
    // frame (an empty image renders as "nothing"), and the failure is
    // visible in the log instead of being swallowed.
    qWarning() << "[AI] inference" << QString::fromLatin1(stage) << "failed:" << outcome.error;
    outcome.image = input;
    return outcome;
}

DetectionOutcome runDetectionGuarded(const char* stage,
                                     const std::function<std::vector<AIDetectionResult>(const cv::Mat&)>& call,
                                     const cv::Mat& input)
{
    DetectionOutcome outcome;
    try {
        outcome.results = call(input);
        outcome.ok = true;
        return outcome;
    } catch (const cv::Exception& e) {
        outcome.error = describeException(stage, e);
    } catch (const std::exception& e) {
        outcome.error = describeException(stage, e);
    } catch (...) {
        outcome.error = QStringLiteral("%1: failed with an unknown error")
                            .arg(QString::fromLatin1(stage));
    }
    qWarning() << "[AI] detection" << QString::fromLatin1(stage) << "failed:" << outcome.error;
    outcome.results.clear();
    return outcome;
}

cv::Mat imageOrEmpty(const InferenceOutcome& outcome)
{
    // Empty on failure — see the header: this is what turns a contained
    // exception into the `aiproc.inference_failed` the user is shown.
    if (!outcome.ok)
        return {};
    return outcome.image;
}

} // namespace ai
