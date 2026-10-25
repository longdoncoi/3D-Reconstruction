#ifndef AIINFERENCEGUARD_H
#define AIINFERENCEGUARD_H

#include <QString>
#include <functional>
#include <vector>

#include "IAIService.h"   // AIDetectionResult
#include "Global.h"        // APP_EXPORT

namespace cv { class Mat; }

namespace ai {

/**
 * @brief The outcome of a guarded inference call.
 */
struct InferenceOutcome {
    bool    ok    = false;  ///< true when the call completed without throwing
    QString error;          ///< empty when ok
    cv::Mat image;          ///< the produced image (the INPUT when the call failed)
};

/**
 * @brief The outcome of a guarded detection call.
 */
struct DetectionOutcome {
    bool                            ok      = false;
    QString                         error;
    std::vector<AIDetectionResult>  results;
};

/**
 * @brief Runs an inference call and CONTAINS any exception it throws.
 *
 * ONNX Runtime (`Ort::Exception`), OpenCV (`cv::Exception`) and the standard
 * allocator (`std::bad_alloc` when the 227 MB session does not fit) all
 * throw.  An exception that escapes a Qt slot or a `QThread::run()` reaches
 * `std::terminate()` and **aborts the whole process** (0xC0000409): the
 * application dies instead of reporting one failed request.
 *
 * The model-LOAD path already contains its exceptions (`AIProcessor::
 * loadDetectionModel` and friends).  The model-INFERENCE path did not — and
 * inference is the path that runs on a background thread
 * (`VideoTrackerThread`) as well as on the GUI thread
 * (`AIProcessorPlugin::onDetection` / `onSegmentation`), where an escaping
 * exception is fatal in both cases.
 *
 * This helper is the single place where the inference path is contained, so
 * the guard cannot be applied to one entry point and forgotten on the next
 * one.  On failure the caller receives the **input image unchanged** (an
 * empty frame would look like a broken render) and the failure is logged.
 */
APP_EXPORT InferenceOutcome runInferenceGuarded(const char* stage,
                                                const std::function<cv::Mat(const cv::Mat&)>& call,
                                                const cv::Mat& input);

/// Same contract as runInferenceGuarded() for a detection call that returns a
/// list of boxes instead of an image.
APP_EXPORT DetectionOutcome runDetectionGuarded(const char* stage,
                                                const std::function<std::vector<AIDetectionResult>(const cv::Mat&)>& call,
                                                const cv::Mat& input);

/**
 * @brief Translates a guarded outcome into the image the SERVICE returns.
 *
 * `runInferenceGuarded()` hands the INPUT back on failure: that is right for a
 * renderer that would otherwise paint a blank frame, and it keeps the guard's
 * own contract simple. But the plugin-facing `IAIService` contract is the
 * opposite one — "the rendered image, or EMPTY on error" — and the consumer
 * reads it literally: `AIProcessorPlugin::finishInference()` reports
 * `aiproc.inference_failed` only when the image is empty. Returning the input
 * on failure therefore made a failed inference indistinguishable from a
 * successful one that changed nothing: the user got the untouched frame with
 * no error at all.
 *
 * This is the ONE place where the two contracts are reconciled, so no entry
 * point can forget it: a failed outcome becomes an empty image and a
 * successful one is passed through unchanged.
 */
APP_EXPORT cv::Mat imageOrEmpty(const InferenceOutcome& outcome);

} // namespace ai

#endif // AIINFERENCEGUARD_H
