#include <QtTest>

#include <QDialog>
#include <QFile>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QSignalSpy>
#include <QTemporaryDir>

#include <opencv2/core.hpp>

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
class TestPluginWidgets : public QObject {
    Q_OBJECT

private slots:
    void chatImageViewerShowsASealedAttachment();
    void chatImageViewerSurvivesAMissingFile();
    void chatImageViewerClosesOnAClick();
    void videoTrackerReportsAnUnreadableVideo();
    void videoTrackerStopsCleanly();
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

QTEST_MAIN(TestPluginWidgets)
#include "TestPluginWidgets.moc"
