#include <QtTest>
#include <QTemporaryFile>
#include <QSignalSpy>
#include <QThread>

#include "IAIModelLoadService.h"
#include "IAIModelAdmin.h"
#include "AIModelLoadService.h"

#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <atomic>

/**
 * @brief Contract test for the AppCore-owned async model loader.
 *
 * The loader is the component that made plugin unload safe: it must own the
 * worker lifetime, skip missing paths, report progress and finish, and let a
 * cancel request reach the worker between models.
 */
// The loader talks to the AppCore-internal IAIModelAdmin seam, not to the
// plugin-facing IAIService: the synchronous loaders are deliberately not part
// of the plugin contract.
class FakeAIService : public IAIModelAdmin {
public:
    bool loadDetectionModel(const QString &) override {
        ++loadCalls;
        loadThread.store(QThread::currentThread());
        if (throwNonStandard) throw 42;             // exercises catch (...)
        if (throwDetection) throw std::runtime_error("detection boom");
        QThread::msleep(100); // give the test time to cancel mid-load
        return !failDetection;
    }
    bool loadSegmentationModel(const QString &) override {
        ++segmentationCalls;
        if (throwSegmentation) throw std::runtime_error("segmentation boom");
        return !failSegmentation;
    }
    bool loadTrackingModel(const QString &) override { return true; }

    int loadCalls = 0;
    int segmentationCalls = 0;
    std::atomic<QThread *> loadThread{nullptr};
    bool failDetection = false;
    bool failSegmentation = false;
    bool throwDetection = false;
    bool throwSegmentation = false;
    bool throwNonStandard = false;
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
    void throwingLoaderReportsFailureAndClearsRunning();
    void throwingLoaderAfterFailureKeepsBothErrors();
    void nonStandardExceptionIsCaught();
    void loadRunsOffTheCallingThread();

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

void TestAIModelLoadService::throwingLoaderReportsFailureAndClearsRunning()
{
    FakeAIService ai;
    ai.throwDetection = true;
    AIModelLoadService service(&ai);

    QTemporaryFile detection;
    QVERIFY(detection.open());
    detection.write("stub");
    detection.close();

    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(detection.fileName(), QString(), QString());
    QVERIFY(waitForSignal(finished));

    // An exception from a loader must surface as a normal failure and, above
    // all, must reset m_running — otherwise every later load is rejected and
    // the service looks busy forever.
    const QList<QVariant> args = finished.first();
    QCOMPARE(args.at(0).toBool(), false);
    QVERIFY(args.at(1).toString().contains(QStringLiteral("Model load failed")));
    QVERIFY(!service.isLoading());
}

void TestAIModelLoadService::throwingLoaderAfterFailureKeepsBothErrors()
{
    FakeAIService ai;
    ai.failDetection = true;      // sets the error string first…
    ai.throwSegmentation = true;  // …then a later model throws
    AIModelLoadService service(&ai);

    QTemporaryFile detection;
    QVERIFY(detection.open());
    detection.write("stub");
    detection.close();
    QTemporaryFile segmentation;
    QVERIFY(segmentation.open());
    segmentation.write("stub");
    segmentation.close();

    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(detection.fileName(), segmentation.fileName(), QString());
    QVERIFY(waitForSignal(finished));

    // The earlier failure detail must not be dropped by the exception.
    const QString error = finished.first().at(1).toString();
    QVERIFY(error.contains(QStringLiteral("detection")));
    QVERIFY(error.contains(QStringLiteral("Model load failed")));
    QVERIFY(error.contains(QStringLiteral("; ")));
    QVERIFY(!service.isLoading());
}

void TestAIModelLoadService::nonStandardExceptionIsCaught()
{
    FakeAIService ai;
    ai.throwNonStandard = true;
    AIModelLoadService service(&ai);

    QTemporaryFile detection;
    QVERIFY(detection.open());
    detection.write("stub");
    detection.close();

    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    service.loadModelsAsync(detection.fileName(), QString(), QString());
    QVERIFY(waitForSignal(finished));

    const QList<QVariant> args = finished.first();
    QCOMPARE(args.at(0).toBool(), false);
    QVERIFY(args.at(1).toString().contains(QStringLiteral("unknown error")));
    QVERIFY(!service.isLoading());
}

void TestAIModelLoadService::loadRunsOffTheCallingThread()
{
    FakeAIService ai;
    AIModelLoadService service(&ai);

    QTemporaryFile detection;
    QVERIFY(detection.open());
    detection.write("stub");
    detection.close();

    QSignalSpy finished(&service, &IAIModelLoadService::finished);

    QThread *const caller = QThread::currentThread();
    service.loadModelsAsync(detection.fileName(), QString(), QString());
    QVERIFY(waitForSignal(finished));

    // The contract that makes a GUI-triggered load safe: the loader never
    // runs on the caller's thread. `QtConcurrent::run` does not guarantee
    // this (a pool with no free worker runs the task inline), which is why
    // the service uses blocking::run's dedicated thread.
    QVERIFY(ai.loadThread.load() != nullptr);
    QVERIFY(ai.loadThread.load() != caller);
    QVERIFY(!service.isLoading());
}

QTEST_GUILESS_MAIN(TestAIModelLoadService)
#include "TestAIModelLoadService.moc"
