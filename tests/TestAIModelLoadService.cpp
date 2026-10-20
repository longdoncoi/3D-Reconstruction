#include <QtTest>
#include <QTemporaryFile>
#include <QSignalSpy>
#include <QThread>

#include "IAIModelLoadService.h"
#include "IAIService.h"
#include "AIModelLoadService.h"

#include <opencv2/core.hpp>

#include <memory>

/**
 * @brief Contract test for the AppCore-owned async model loader.
 *
 * The loader is the component that made plugin unload safe: it must own the
 * worker lifetime, skip missing paths, report progress and finish, and let a
 * cancel request reach the worker between models.
 */
class FakeAIService : public IAIService {
public:
    bool loadDetectionModel(const QString &) override {
        ++loadCalls;
        QThread::msleep(100); // give the test time to cancel mid-load
        return !failDetection;
    }
    bool loadSegmentationModel(const QString &) override {
        ++segmentationCalls;
        return !failSegmentation;
    }
    bool loadTrackingModel(const QString &) override { return true; }
    bool isDetectionReady() const override { return true; }
    bool isSegmentationReady() const override { return true; }
    bool isTrackingReady() const override { return true; }
    std::vector<AIDetectionResult> detectObjects(const cv::Mat &) override { return {}; }
    cv::Mat runDetection(const cv::Mat &) override { return {}; }
    cv::Mat runSegmentation(const cv::Mat &) override { return {}; }
    cv::Mat runTracking(const cv::Mat &) override { return {}; }
    void resetTrackingState() override {}
    void setConfidenceThreshold(float) override {}
    float confidenceThreshold() const override { return 0.5f; }

    int loadCalls = 0;
    int segmentationCalls = 0;
    bool failDetection = false;
    bool failSegmentation = false;
};

class TestAIModelLoadService : public QObject {
    Q_OBJECT

private slots:
    void missingPathsAreSkippedAndFinishCleanly();
    void failedModelReportsError();
    void cancelIsObservedByTheWorker();
    void nullAiServiceIsRejected();
    void concurrentLoadIsRejected();
    void destructionCancelsRunningWorker();

private:
    static bool waitForSignal(QSignalSpy &spy, int timeoutMs = 3000)
    {
        return spy.wait(timeoutMs);
    }
};

void TestAIModelLoadService::missingPathsAreSkippedAndFinishCleanly()
{
    FakeAIService ai;
    AIModelLoadService service(&ai);

    QSignalSpy stages(&service, &IAIModelLoadService::stageChanged);
    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(QString(), QString(), QString());
    QVERIFY(waitForSignal(finished));
    QCOMPARE(stages.count(), 0);
    QCOMPARE(ai.loadCalls, 0);
    QVERIFY(!service.isLoading());
}

void TestAIModelLoadService::failedModelReportsError()
{
    FakeAIService ai;
    ai.failDetection = true;
    ai.failSegmentation = true;
    AIModelLoadService service(&ai);

    QTemporaryFile detection;
    QVERIFY(detection.open());
    detection.write("stub");
    detection.close();
    QTemporaryFile segmentation;
    QVERIFY(segmentation.open());
    segmentation.write("stub");
    segmentation.close();

    QSignalSpy stages(&service, &IAIModelLoadService::stageChanged);
    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(detection.fileName(), segmentation.fileName(), QString());
    QVERIFY(waitForSignal(finished));
    QCOMPARE(stages.count(), 2);
    QCOMPARE(stages.at(0).first().toString(), QStringLiteral("detection"));
    QCOMPARE(stages.at(1).first().toString(), QStringLiteral("segmentation"));
    QCOMPARE(ai.loadCalls, 1);
    QCOMPARE(ai.segmentationCalls, 1);

    const QList<QVariant> args = finished.first();
    QCOMPARE(args.at(0).toBool(), false);
    const QString error = args.at(1).toString();
    QVERIFY(error.contains(QStringLiteral("detection")));
    QVERIFY(error.contains(QStringLiteral("segmentation")));
    QVERIFY(error.contains(QStringLiteral("; ")));
}

void TestAIModelLoadService::destructionCancelsRunningWorker()
{
    FakeAIService ai;
    auto service = std::make_unique<AIModelLoadService>(&ai);

    QTemporaryFile model;
    QVERIFY(model.open());
    model.write("stub");
    model.close();

    QSignalSpy stages(service.get(), &IAIModelLoadService::stageChanged);
    service->loadModelsAsync(model.fileName(), QString(), QString());
    QVERIFY(stages.wait(3000));

    // Destructor must cancel and join the private pool instead of letting a
    // worker outlive the object and emit into freed memory.
    service.reset();
    QCOMPARE(ai.loadCalls, 1);
}

void TestAIModelLoadService::cancelIsObservedByTheWorker()
{
    FakeAIService ai;
    AIModelLoadService service(&ai);

    QTemporaryFile model;
    QVERIFY(model.open());
    model.write("stub");
    model.close();

    QSignalSpy stages(&service, &IAIModelLoadService::stageChanged);
    QSignalSpy cancelled(&service, &IAIModelLoadService::cancelled);

    service.loadModelsAsync(model.fileName(), QString(), QString());
    QVERIFY(waitForSignal(stages));

    service.cancel();
    QVERIFY(waitForSignal(cancelled));
    QVERIFY(!service.isLoading());
}

void TestAIModelLoadService::nullAiServiceIsRejected()
{
    AIModelLoadService service(nullptr);
    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(QStringLiteral("x.onnx"), QString(), QString());

    // A missing IAIService is a wiring bug: the call must be a no-op instead
    // of queueing a worker that dereferences nullptr.
    QVERIFY(!finished.wait(100));
    QVERIFY(!service.isLoading());
}

void TestAIModelLoadService::concurrentLoadIsRejected()
{
    FakeAIService ai;
    AIModelLoadService service(&ai);

    QTemporaryFile model;
    QVERIFY(model.open());
    model.write("stub");
    model.close();

    QSignalSpy stages(&service, &IAIModelLoadService::stageChanged);
    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(model.fileName(), QString(), QString());
    QVERIFY(stages.wait(3000));
    QVERIFY(service.isLoading());

    // A second request while a load is running is ignored, not queued.
    service.loadModelsAsync(model.fileName(), QString(), QString());

    QVERIFY(finished.wait(3000));
    QCOMPARE(ai.loadCalls, 1);
    QVERIFY(!service.isLoading());
}

QTEST_GUILESS_MAIN(TestAIModelLoadService)
#include "TestAIModelLoadService.moc"
