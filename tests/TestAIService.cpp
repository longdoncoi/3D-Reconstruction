#include <QtTest>
#include <QDir>
#include <QFileInfo>
#include <QTemporaryDir>

#include "AIService.h"
#include "AIInferenceGuard.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

/**
 * @brief The AI service against the REAL ONNX models.
 *
 * The service is a thin wrapper over AIProcessor, but the wrapper IS the
 * plugin-facing contract: the readiness flags, the threshold and the
 * "a missing model is refused" behaviour. Measuring it needs a real model —
 * a mock would only measure the mock.
 *
 * The models are the ones the training pipeline publishes
 * (AIComputerVision/Models/*.onnx, Git LFS). When they are not checked out
 * the test SKIPS: a checkout without `git lfs pull` must not fail the build.
 * It also must not pretend the path was covered — the coverage floor of
 * aiservice.cpp is only met on a runner that HAS the models (the C++ CI
 * enables `lfs: true`).
 */
class TestAIService : public QObject {
    Q_OBJECT

private slots:
    void aMissingModelIsRefused();
    void theDetectionModelLoadsAndDetects();
    void theSegmentationModelLoadsAndSegments();
    void theConfidenceThresholdRoundTrips();
    void anInferenceExceptionIsContained();
    void aDetectionExceptionIsContained();
    void anOpenCvExceptionIsContained();
    void anInferenceWithoutAModelReturnsEmpty();
    void aFailedOutcomeBecomesAnEmptyImage();

private:
    /// The coverage exporter cannot host a real ONNX session: its
    /// instrumentation multiplies every allocation and the session creation
    /// dies with std::bad_alloc. The coverage gate therefore sets this
    /// variable, and the model-loading cases skip — the service's OWN lines
    /// (the delegation, the readiness flags, the refusal of a missing model)
    /// are still measured, and the real inference is proven by the normal
    /// CTest run, where no exporter is involved.
    static bool modelsDisabledByTheExporter()
    {
        return !qEnvironmentVariableIsEmpty("AI_SERVICE_NO_MODELS");
    }

    /// The published model file, or an empty string when it is not checked out.
    static QString modelPath(const QString &name)
    {
        const QString path = QStringLiteral(AI_MODELS_DIR) + "/" + name;
        // An LFS pointer is a ~130 byte text file, not a model.
        if (!QFileInfo::exists(path) || QFileInfo(path).size() < 1024 * 1024)
            return {};
        return path;
    }

    /// A synthetic image with a few shapes: the model does not have to detect
    /// anything, the pipeline must simply run.
    static cv::Mat syntheticFrame()
    {
        cv::Mat image(360, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::rectangle(image, cv::Rect(120, 90, 180, 140), cv::Scalar(220, 60, 40), -1);
        cv::circle(image, cv::Point(430, 210), 60, cv::Scalar(40, 200, 90), -1);
        return image;
    }
};

void TestAIService::aMissingModelIsRefused()
{
    AIService service;
    // A path that does not exist must be refused, and the service must NOT
    // report itself ready: a plugin asking "is the model loaded?" gets the
    // truth.
    QVERIFY(!service.loadDetectionModel(QStringLiteral("C:/definitely/missing.onnx")));
    QVERIFY(!service.isDetectionReady());
    QVERIFY(!service.loadSegmentationModel(QStringLiteral("C:/definitely/missing.onnx")));
    QVERIFY(!service.isSegmentationReady());
    QVERIFY(!service.loadTrackingModel(QStringLiteral("C:/definitely/missing.onnx")));
    QVERIFY(!service.isTrackingReady());

    // The box API reports the same truth: no model ⇒ the inference did not
    // run (ok == false), not "detected nothing".
    QVERIFY(!service.detectObjects(syntheticFrame()).ok);
}

void TestAIService::theDetectionModelLoadsAndDetects()
{
    if (modelsDisabledByTheExporter())
        QSKIP("the coverage exporter cannot host a real ONNX session");
    const QString path = modelPath(QStringLiteral("yolo11n.onnx"));
    if (path.isEmpty())
        QSKIP("the ONNX models are not checked out (run `git lfs pull`)");

    AIService service;
    QVERIFY2(service.loadDetectionModel(path), "the detection model did not load");
    QVERIFY(service.isDetectionReady());

    // The whole pipeline: preprocess → inference → postprocess.
    const cv::Mat frame = syntheticFrame();
    const AIDetectionOutcome objects = service.detectObjects(frame);
    QVERIFY2(objects.ok, "a loaded model must report the inference as successful");
    for (const auto &object : objects.objects) {
        QVERIFY(object.confidence >= 0.0f);
        QVERIFY(object.confidence <= 1.0f);
        QVERIFY(object.x >= 0);
    }

    const cv::Mat rendered = service.runDetection(frame);
    QVERIFY2(!rendered.empty(), "the detection produced no output");
}

void TestAIService::theSegmentationModelLoadsAndSegments()
{
    if (modelsDisabledByTheExporter())
        QSKIP("the coverage exporter cannot host a real ONNX session");
    const QString path = modelPath(QStringLiteral("yolo11n-seg.onnx"));
    if (path.isEmpty())
        QSKIP("the ONNX models are not checked out (run `git lfs pull`)");

    AIService service;
    QVERIFY2(service.loadSegmentationModel(path), "the segmentation model did not load");
    QVERIFY(service.isSegmentationReady());

    const cv::Mat frame = syntheticFrame();
    const cv::Mat segmented = service.runSegmentation(frame);
    QVERIFY2(!segmented.empty(), "the segmentation produced no output");
}

void TestAIService::theConfidenceThresholdRoundTrips()
{
    AIService service;
    const float before = service.confidenceThreshold();
    service.setConfidenceThreshold(0.75f);
    QCOMPARE(service.confidenceThreshold(), 0.75f);
    service.setConfidenceThreshold(before);
    QCOMPARE(service.confidenceThreshold(), before);
}

void TestAIService::anInferenceExceptionIsContained()
{
    // ONNX Runtime and OpenCV throw. An exception that escapes a Qt slot
    // (the GUI thread runs the detection/segmentation) or a QThread::run()
    // (the tracking thread) reaches std::terminate() and ABORTS THE WHOLE
    // PROCESS (0xC0000409) — the application dies instead of reporting one
    // failed request. The guard is the single place where that is prevented.
    const cv::Mat frame = syntheticFrame();

    bool threw = false;
    const ai::InferenceOutcome outcome = ai::runInferenceGuarded(
        "test",
        [&threw](const cv::Mat&) -> cv::Mat {
            threw = true;
            throw std::runtime_error("the inference backend failed");
        },
        frame);

    QVERIFY(threw);
    QVERIFY2(!outcome.ok, "a throwing inference must be reported as a failure");
    QVERIFY(!outcome.error.isEmpty());
    // The caller gets the INPUT back: an empty frame would render as
    // "nothing", and garbage would be worse.
    QCOMPARE(outcome.image.size(), frame.size());
    QCOMPARE(outcome.image.type(), frame.type());
}

void TestAIService::aDetectionExceptionIsContained()
{
    // The same contract for the detection entry point, which returns a list
    // of boxes instead of an image.
    const cv::Mat frame = syntheticFrame();

    bool threw = false;
    const ai::DetectionOutcome outcome = ai::runDetectionGuarded(
        "test",
        [&threw](const cv::Mat&) -> std::vector<AIDetectionResult> {
            threw = true;
            throw std::runtime_error("the inference backend failed");
        },
        frame);

    QVERIFY(threw);
    QVERIFY(!outcome.ok);
    QVERIFY(!outcome.error.isEmpty());
    QVERIFY(outcome.results.empty());
}

void TestAIService::anOpenCvExceptionIsContained()
{
    // OpenCV throws its own exception type (cv::Exception) — a different
    // catch branch than the standard one, and the one the image pipeline
    // actually raises.
    const cv::Mat frame = syntheticFrame();

    bool threw = false;
    const ai::InferenceOutcome outcome = ai::runInferenceGuarded(
        "test",
        [&threw](const cv::Mat&) -> cv::Mat {
            threw = true;
            throw cv::Exception(cv::Error::StsAssert, "boom", "fn", "file", 1);
        },
        frame);

    QVERIFY(threw);
    QVERIFY(!outcome.ok);
    QVERIFY(outcome.error.contains(QStringLiteral("boom")));
    QCOMPARE(outcome.image.size(), frame.size());
}

void TestAIService::anInferenceWithoutAModelReturnsEmpty()
{
    // The plugin-facing contract (IAIService.h) is "the rendered image, or
    // empty on error". AIProcessor returns the INPUT when no model is loaded,
    // which reads exactly like a successful run that detected nothing — so the
    // service has to refuse instead of passing it through. Without this, the
    // plugin renders the untouched frame and reports success.
    AIService service;
    const cv::Mat frame = syntheticFrame();

    QVERIFY2(service.runDetection(frame).empty(),
             "detection without a model must be an error, not the input frame");
    QVERIFY2(service.runSegmentation(frame).empty(),
             "segmentation without a model must be an error, not the input frame");
    QVERIFY2(service.runTracking(frame).empty(),
             "tracking without a model must be an error, not the input frame");
}

void TestAIService::aFailedOutcomeBecomesAnEmptyImage()
{
    // runInferenceGuarded() returns the INPUT on failure (a renderer that
    // ignores the flag must not paint a black frame). The service boundary is
    // where that becomes the empty image the plugin reads as a failure.
    const cv::Mat frame = syntheticFrame();

    ai::InferenceOutcome failed;
    failed.ok = false;
    failed.error = QStringLiteral("boom");
    failed.image = frame;   // the guard hands the input back
    QVERIFY2(ai::imageOrEmpty(failed).empty(),
             "a failed outcome must not be rendered as the input frame");

    ai::InferenceOutcome succeeded;
    succeeded.ok = true;
    succeeded.image = frame.clone();
    QCOMPARE(ai::imageOrEmpty(succeeded).size(), frame.size());
}

QTEST_GUILESS_MAIN(TestAIService)
#include "TestAIService.moc"
