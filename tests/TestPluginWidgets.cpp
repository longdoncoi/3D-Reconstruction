#include <QtTest>

#include <QDialog>
#include <QFile>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QSignalSpy>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "AtRestCipher.h"
#include "ChatImageViewer.h"
#include "FileUtilities.h"
#include "VideoTrackerThread.h"
#include "mock/PluginContextMock.h"

/**
 * @brief The last two plugin translation units without a test.
 *
 * `ChatImageViewer` shows an attachment that is sealed at rest (the file on
 * disk is a ciphertext — the viewer has to unseal it before it can build a
 * pixmap), and `VideoTrackerThread` runs the tracking loop on its own thread
 * and has to report an unreadable video instead of spinning.
 */
namespace {

/// Writes a tiny synthetic clip so the tracking loop actually reaches the
/// inference. Returns false when this OpenCV build has no video backend.
bool writeSyntheticClip(const QString& path)
{
    cv::VideoWriter writer(path.toStdString(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           10.0, cv::Size(64, 64));
    if (!writer.isOpened()) return false;
    for (int i = 0; i < 5; ++i) {
        cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(0, 0, 255));
        writer.write(frame);
    }
    writer.release();
    return true;
}

/// The clip must be READABLE, not only writable: a codec that writes an
/// unreadable file would make the tracking loop bail out before the
/// inference, and the test would prove nothing.
bool clipIsReadable(const QString& path)
{
    cv::VideoCapture cap(path.toStdString());
    if (!cap.isOpened()) return false;
    cv::Mat frame;
    return cap.read(frame) && !frame.empty();
}

} // namespace

class TestPluginWidgets : public QObject {
    Q_OBJECT

private slots:
    void chatImageViewerShowsASealedAttachment();
    void chatImageViewerSurvivesAMissingFile();
    void chatImageViewerClosesOnAClick();
    void videoTrackerReportsAnUnreadableVideo();
    void videoTrackerStopsCleanly();
    void videoTrackerContainsAThrowingInference();
    void videoTrackerReportsAFailedInference();
    void videoTrackerTerminatesAThreadThatCannotStopCooperatively();
};

void TestPluginWidgets::chatImageViewerShowsASealedAttachment()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    // A real PNG, sealed the way the attachment pipeline stores it.
    QPixmap source(32, 24);
    source.fill(Qt::blue);
    QByteArray png;
    QBuffer buffer(&png);
    QVERIFY(buffer.open(QIODevice::WriteOnly));
    QVERIFY(source.save(&buffer, "PNG"));
    buffer.close();

    const QString context = QString::fromLatin1(FileUtilities::AttachmentContext);
    const QString path = dir.filePath(QStringLiteral("attachment.bin"));
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write(AtRestCipher::seal(png, context));
    file.close();

    ChatImageViewer viewer(path);
    const QList<QLabel *> labels = viewer.findChildren<QLabel *>();
    QVERIFY(!labels.isEmpty());
    bool hasImage = false;
    for (QLabel *label : labels) {
        if (!label->pixmap().isNull()) {
            hasImage = true;
            break;
        }
    }
    QVERIFY2(hasImage, "the sealed attachment was not unsealed into a pixmap");
}

void TestPluginWidgets::chatImageViewerSurvivesAMissingFile()
{
    // A missing attachment must not crash the viewer: it simply has no image.
    ChatImageViewer viewer(QStringLiteral("no-such-attachment.bin"));
    for (QLabel *label : viewer.findChildren<QLabel *>()) {
        QVERIFY(label->pixmap().isNull());
    }
}

void TestPluginWidgets::chatImageViewerClosesOnAClick()
{
    ChatImageViewer viewer(QStringLiteral("no-such-attachment.bin"));
    viewer.show();

    // A click anywhere closes the viewer (the event filter accepts it).
    QMouseEvent press(QEvent::MouseButtonPress, QPointF(5, 5),
                      QPointF(5, 5), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&viewer, &press);
    QCOMPARE(viewer.result(), int(QDialog::Accepted));
}

void TestPluginWidgets::videoTrackerReportsAnUnreadableVideo()
{
    PluginMock::AIService aiService;
    VideoTrackerThread thread(&aiService, QStringLiteral("no-such-video.mp4"));

    QSignalSpy errors(&thread, &VideoTrackerThread::errorOccurred);
    thread.start();
    QVERIFY(errors.wait(10000));
    QVERIFY(errors.first().first().toString().contains(QStringLiteral("Could not open")));

    QVERIFY(thread.stopAndWait(5000));
}

void TestPluginWidgets::videoTrackerStopsCleanly()
{
    PluginMock::AIService aiService;
    VideoTrackerThread thread(&aiService, QStringLiteral("no-such-video.mp4"));

    // pause()/resume()/stop() must be safe before and after the loop ended.
    thread.pause();
    thread.resume();
    thread.start();
    QVERIFY(thread.stopAndWait(5000));
    thread.stop();
}

void TestPluginWidgets::videoTrackerContainsAThrowingInference()
{
    // A thread entry point must contain the exceptions of its body. An
    // exception that leaves QThread::run() reaches std::terminate() and
    // ABORTS THE WHOLE PROCESS (0xC0000409) — the app dies instead of
    // reporting a failed tracking session.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString clip = dir.filePath(QStringLiteral("clip.avi"));
    if (!writeSyntheticClip(clip) || !clipIsReadable(clip)) {
        QSKIP("this OpenCV build has no usable video backend — the clip cannot be written");
    }

    PluginMock::ThrowingAIService aiService;
    VideoTrackerThread thread(&aiService, clip);

    QSignalSpy errors(&thread, &VideoTrackerThread::errorOccurred);
    QSignalSpy finished(&thread, &VideoTrackerThread::finishedTracking);
    thread.start();

    QVERIFY(errors.wait(15000));
    const QString message = errors.first().first().toString();
    QVERIFY2(message.contains(QStringLiteral("Tracking failed")),
             qPrintable(QStringLiteral("unexpected error message: %1").arg(message)));
    // The shell must be able to leave the "tracking" state: the thread
    // reports the failure AND ends the session.
    QTRY_VERIFY2(finished.count() > 0,
                 qPrintable(QStringLiteral("finishedTracking was not emitted "
                                           "(%1 error(s) reported)")
                                .arg(errors.count())));

    QVERIFY(thread.stopAndWait(5000));
}

void TestPluginWidgets::videoTrackerReportsAFailedInference()
{
    // The PRODUCTION failure shape. AIService contains the inference exception
    // (ai::runInferenceGuarded) and reports it as an EMPTY image — nothing is
    // thrown, so the thread's own barrier never fires. Before the loop checked
    // this, every frame was forwarded as if it had been tracked: the view kept
    // the last picture, the UI stayed in Tracking mode and the session ended
    // with the "finished" message. A silent no-op that looked like success.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString clip = dir.filePath(QStringLiteral("clip.avi"));
    if (!writeSyntheticClip(clip) || !clipIsReadable(clip)) {
        QSKIP("this OpenCV build has no usable video backend — the clip cannot be written");
    }

    // The base mock returns an EMPTY image from runTracking() when asked to.
    PluginMock::AIService aiService;
    aiService.returnEmpty = true;
    VideoTrackerThread thread(&aiService, clip);

    QSignalSpy errors(&thread, &VideoTrackerThread::errorOccurred);
    QSignalSpy finished(&thread, &VideoTrackerThread::finishedTracking);
    thread.start();

    QVERIFY2(errors.wait(15000), "the failed inference was never reported");
    const QString message = errors.first().first().toString();
    QVERIFY2(message.contains(QStringLiteral("no result")),
             qPrintable(QStringLiteral("unexpected error message: %1").arg(message)));
    // The session still ends, so the shell can leave Tracking mode.
    QTRY_VERIFY2(finished.count() > 0, "the session never ended");

    QVERIFY(thread.stopAndWait(5000));
}

void TestPluginWidgets::videoTrackerTerminatesAThreadThatCannotStopCooperatively()
{
    // The last-resort path of the join. A thread stuck inside the inference
    // backend cannot observe m_stopRequested: the cooperative stop waits out
    // its timeout and MUST fall back to terminate(). Without the fallback the
    // thread outlives the plugin and calls into a destroyed IAIService — the
    // exact use-after-free the bounded wait exists to prevent.
    //
    // The join budget here is deliberately shorter than the inference, so the
    // cooperative path cannot succeed. The destructor joins again with the
    // default 15 s, so a failed assertion still tears the thread down.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString clip = dir.filePath(QStringLiteral("clip.avi"));
    if (!writeSyntheticClip(clip) || !clipIsReadable(clip)) {
        QSKIP("this OpenCV build has no usable video backend — the clip cannot be written");
    }

    PluginMock::AIService aiService;
    aiService.inferenceDelayMs = 3000; // far longer than the 100 ms join budget
    VideoTrackerThread thread(&aiService, clip);
    thread.start();

    // Give the loop time to enter runTracking() and park there.
    QTest::qWait(200);
    QVERIFY(thread.isRunning());

    QVERIFY2(!thread.stopAndWait(100),
             "stopAndWait() reported a cooperative stop while the inference was still running");
    QVERIFY2(!thread.isRunning(),
             "the thread survived the terminate() fallback — it would outlive its IAIService");
}

QTEST_MAIN(TestPluginWidgets)
#include "TestPluginWidgets.moc"
