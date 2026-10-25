#include <QtTest>

#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QTemporaryDir>
#include <QWidget>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdexcept>

#include "AIPredictionLogService.h"
#include "AIAttachmentPreviewFactory.h"
#include "IAppPaths.h"
#include "IMailService.h"
#include "MailInboxItemFactory.h"
#include "MailMessageFormatter.h"

/**
 * @brief Contract of the plugin helpers that carry real logic.
 *
 * The shipped plugins are GUI shells, but the logic behind them is not:
 * the mail preview renderer (quoted-printable decoding, HTML detection,
 * quote rendering, escaping), the inbox row factory, the attachment preview
 * factory and the prediction logger. They were the last plugin files without
 * a single executed line.
 */
class TestPluginHelpers : public QObject {
    Q_OBJECT

private slots:
    // ── MailMessageFormatter ────────────────────────────────────────────────
    void senderNameIsExtractedFromTheFromHeader();
    void previewRendersTheHeaderAndEscapesTheSubject();
    void previewFallsBackToTheEmptyBodyText();
    void previewDecodesQuotedPrintable();
    void previewRendersQuotedLinesAsQuotes();
    void previewKeepsAnHtmlBodyAsHtml();

    // ── MailInboxItemFactory ────────────────────────────────────────────────
    void inboxItemCarriesTheUidAndTheUnreadMarker();

    // ── AIAttachmentPreviewFactory ──────────────────────────────────────────
    void attachmentPreviewHasARemoveButton();

    // ── AIPredictionLogService ──────────────────────────────────────────────
    void predictionLoggerIgnoresANullPathProvider();
    void predictionLoggerWritesTheResultImage();
    void predictionLoggerContainsAThrowingWriter();
    void predictionLoggerReportsAnUnwritableDirectory();
};

namespace {

/// Minimal IAppPaths stub: the prediction logger only needs predictDir().
class StubAppPaths : public IAppPaths {
public:
    explicit StubAppPaths(QString root) : m_root(std::move(root)) {}
    QString appDir() const override { return m_root; }
    QString projectRootDir() const override { return m_root; }
    QString dataRootDir() const override { return m_root; }
    QString configPath() const override { return m_root + "/Config.ini"; }
    QString logsDir() const override { return m_root + "/logs"; }
    QString modelsDir() const override { return m_root + "/Models"; }
    QString predictDir(const QString &type) const override { return m_root + "/" + type; }
    QString aiAssistantDir() const override { return m_root + "/AIAssistant"; }
    QString aiComputerVisionDir() const override { return m_root + "/AIComputerVision"; }
    QString uploadDir() const override { return m_root + "/Upload"; }
    QString thumbnailsDir() const override { return m_root + "/Thumbnails"; }
    QString pluginsDir() const override { return m_root + "/plugins"; }

private:
    QString m_root;
};

MailMessage makeMessage(const QString &subject, const QString &body, bool isRead = true)
{
    MailMessage message;
    message.uid = "uid-1";
    message.from = "\"Alice\" <alice@example.com>";
    message.to = "bob@example.com";
    message.subject = subject;
    message.htmlBody = body;
    message.isRead = isRead;
    message.date = QDateTime(QDate(2026, 10, 20), QTime(9, 30));
    return message;
}

} // namespace

void TestPluginHelpers::senderNameIsExtractedFromTheFromHeader()
{
    QCOMPARE(MailMessageFormatter::senderName("\"Alice\" <alice@example.com>"),
             QStringLiteral("Alice"));
    QCOMPARE(MailMessageFormatter::senderName("Bob <bob@example.com>"),
             QStringLiteral("Bob"));
    // Only an address → the address itself.
    QCOMPARE(MailMessageFormatter::senderName("<carol@example.com>"),
             QStringLiteral("carol@example.com"));
    QCOMPARE(MailMessageFormatter::senderName("  dave@example.com  "),
             QStringLiteral("dave@example.com"));
}

void TestPluginHelpers::previewRendersTheHeaderAndEscapesTheSubject()
{
    const MailMessage message = makeMessage("<script>alert(1)</script>", "Hello");
    const QString html = MailMessageFormatter::previewHtml(message, "empty");

    QVERIFY(html.contains(QStringLiteral("mail-header")));
    QVERIFY(html.contains(QStringLiteral("alice@example.com")));
    QVERIFY(html.contains(QStringLiteral("bob@example.com")));
    // The subject must be escaped — the preview is rendered as HTML.
    QVERIFY(!html.contains(QStringLiteral("<script>")));
    QVERIFY(html.contains(QStringLiteral("&lt;script&gt;")));
}

void TestPluginHelpers::previewFallsBackToTheEmptyBodyText()
{
    const MailMessage message = makeMessage(QStringLiteral("Subject"), QString());
    const QString html = MailMessageFormatter::previewHtml(message, "no body");
    QVERIFY(html.contains(QStringLiteral("no body")));
    QVERIFY(html.contains(QStringLiteral("font-style:italic")));
}

void TestPluginHelpers::previewDecodesQuotedPrintable()
{
    // "=C3=A9" is the quoted-printable encoding of "é".
    const MailMessage message = makeMessage(QStringLiteral("QP"),
                                             QStringLiteral("Xin ch=C3=A0o"));
    const QString html = MailMessageFormatter::previewHtml(message, "empty");
    QVERIFY(html.contains(QStringLiteral("Xin chào")));
    QVERIFY(!html.contains(QStringLiteral("=C3=A0")));
}

void TestPluginHelpers::previewRendersQuotedLinesAsQuotes()
{
    const MailMessage message = makeMessage(
        QStringLiteral("Re"),
        QStringLiteral("Trả lời:\n> câu hỏi cũ\n> dòng hai"));
    const QString html = MailMessageFormatter::previewHtml(message, "empty");
    QVERIFY(html.contains(QStringLiteral("quoted-mail")));
    QVERIFY(html.contains(QStringLiteral("câu hỏi cũ")));
    QVERIFY(html.contains(QStringLiteral("dòng hai")));
}

void TestPluginHelpers::previewKeepsAnHtmlBodyAsHtml()
{
    const MailMessage message = makeMessage(QStringLiteral("HTML"),
                                             QStringLiteral("<b>bold</b>"));
    const QString html = MailMessageFormatter::previewHtml(message, "empty");
    // An HTML body is shown as HTML, not converted to plain text.
    QVERIFY(html.contains(QStringLiteral("<b>bold</b>")));
}

void TestPluginHelpers::inboxItemCarriesTheUidAndTheUnreadMarker()
{
    QListWidget list;
    const MailMessage unread = makeMessage(QStringLiteral("Unread"), "body", false);
    MailInboxItemFactory::addMessageItem(&list, unread, 3, "fallback");

    QCOMPARE(list.count(), 1);
    QListWidgetItem *item = list.item(0);
    QVERIFY(item != nullptr);
    QCOMPARE(item->data(Qt::UserRole).toString(), QStringLiteral("uid-1"));
    QCOMPARE(item->data(Qt::UserRole + 1).toInt(), 3);
    QCOMPARE(item->toolTip(), unread.from);

    QWidget *itemWidget = list.itemWidget(item);
    QVERIFY(itemWidget != nullptr);
    const QList<QLabel *> labels = itemWidget->findChildren<QLabel *>();
    QVERIFY(!labels.isEmpty());
    // An unread message is marked with a dot.
    QVERIFY(labels.first()->text().startsWith(QString::fromUtf8("\xE2\x97\x8F")));

    // A message without a subject uses the fallback text.
    QListWidget second;
    MailInboxItemFactory::addMessageItem(&second, makeMessage(QString(), "body", true), 0,
                                         "fallback subject");
    QListWidgetItem *fallbackItem = second.item(0);
    QVERIFY(fallbackItem != nullptr);
    const QList<QLabel *> fallbackLabels = second.itemWidget(fallbackItem)->findChildren<QLabel *>();
    QVERIFY(!fallbackLabels.isEmpty());
    QCOMPARE(fallbackLabels.first()->text(), QStringLiteral("fallback subject"));
}

void TestPluginHelpers::attachmentPreviewHasARemoveButton()
{
    QWidget parent;
    FileUtilities::AttachmentResult attachment;
    attachment.success = true;
    attachment.thumbnail = QPixmap(80, 80);

    bool removed = false;
    QWidget *preview = AIAttachmentPreviewFactory::create(
        &parent, attachment, [&removed](QWidget *) { removed = true; });
    QVERIFY(preview != nullptr);
    QCOMPARE(preview->parentWidget(), &parent);
    QCOMPARE(preview->size(), QSize(84, 84));

    const QList<QPushButton *> buttons = preview->findChildren<QPushButton *>();
    QCOMPARE(buttons.size(), 1);
    buttons.first()->click();
    QVERIFY(removed);
}

void TestPluginHelpers::predictionLoggerIgnoresANullPathProvider()
{
    cv::Mat image(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    // A null provider must be a no-op — no crash, no file.
    AIPredictionLogService service(nullptr);
    service.logPrediction(QStringLiteral("detection"), QStringLiteral("frame.png"), image);
    QVERIFY(true);
}

void TestPluginHelpers::predictionLoggerWritesTheResultImage()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    StubAppPaths paths(root.path());

    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    // The service owns the worker: the write happens on ITS pool, which the
    // service joins in its destructor — never on a pool nobody owns.
    AIPredictionLogService service(&paths);
    service.logPrediction(QStringLiteral("detection"), QStringLiteral("frame.png"), image);

    // The write happens on the service's own worker: wait for the file.
    const QDir predictRoot(root.path() + "/detection");
    QElapsedTimer timer;
    timer.start();
    QDir found;
    while (timer.elapsed() < 10000) {
        const QStringList days = predictRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        if (!days.isEmpty()) {
            found = QDir(predictRoot.filePath(days.first()));
            if (!found.entryList({QStringLiteral("*.png")}, QDir::Files).isEmpty()) {
                break;
            }
        }
        QTest::qWait(50);
    }

    const QStringList files = found.entryList({QStringLiteral("*.png")}, QDir::Files);
    QCOMPARE(files.size(), 1);
    QVERIFY(files.first().startsWith(QStringLiteral("frame_")));
    // The file must be a readable image with the logged size.
    const cv::Mat written = cv::imread(found.filePath(files.first()).toStdString());
    QCOMPARE(written.cols, 8);
    QCOMPARE(written.rows, 8);
}

void TestPluginHelpers::predictionLoggerContainsAThrowingWriter()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    StubAppPaths paths(root.path());
    const cv::Mat image(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));

    // 1. An encoder that throws a STANDARD exception. cv::imwrite does exactly
    //    this on a failed encode / unwritable destination.
    int calls = 0;
    {
        AIPredictionLogService service(&paths);
        service.setWriterForTesting([&calls](const QString &, const cv::Mat &) -> bool {
            ++calls;
            throw std::runtime_error("the encoder exploded");
        });
        service.logPrediction(QStringLiteral("detection"), QStringLiteral("frame.png"), image);
    }   // the destructor joins the pool: the task cannot outlive the service
    QCOMPARE(calls, 1);

    // 2. A non-standard throw reaches catch (...) — without it the exception
    //    would escape the QRunnable into QThread::run() and abort the process.
    int unknownCalls = 0;
    {
        AIPredictionLogService service(&paths);
        service.setWriterForTesting([&unknownCalls](const QString &, const cv::Mat &) -> bool {
            ++unknownCalls;
            throw 42;
        });
        service.logPrediction(QStringLiteral("segmentation"), QStringLiteral("frame.png"), image);
    }
    QCOMPARE(unknownCalls, 1);

    // 3. A writer that reports failure by return value instead of throwing.
    int refusedCalls = 0;
    {
        AIPredictionLogService service(&paths);
        service.setWriterForTesting([&refusedCalls](const QString &, const cv::Mat &) -> bool {
            ++refusedCalls;
            return false;
        });
        service.logPrediction(QStringLiteral("detection"), QStringLiteral("frame.png"), image);
    }
    QCOMPARE(refusedCalls, 1);

    // 4. A null writer restores the platform encoder: the seam must not be
    //    able to leave the service with no writer at all.
    {
        AIPredictionLogService service(&paths);
        service.setWriterForTesting([](const QString &, const cv::Mat &) { return true; });
        service.setWriterForTesting(nullptr);
        service.logPrediction(QStringLiteral("detection"), QStringLiteral("frame.png"), image);
    }   // the destructor joins the pool: the default writer has finished here

    bool wroteWithTheDefaultWriter = false;
    QDirIterator it(root.path() + QStringLiteral("/detection"),
                    {QStringLiteral("*.png")}, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        wroteWithTheDefaultWriter = true;
        it.next();
    }
    QVERIFY2(wroteWithTheDefaultWriter,
             "setWriterForTesting(nullptr) must fall back to cv::imwrite");
}

QTEST_MAIN(TestPluginHelpers)
#include "TestPluginHelpers.moc"

void TestPluginHelpers::predictionLoggerReportsAnUnwritableDirectory()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    StubAppPaths paths(root.path());

    // A regular file where the destination directory should be. mkpath()
    // cannot succeed under it, and the service must report that instead of
    // writing somewhere else (or crashing on a half-built path).
    const QString blockerPath = root.path() + QStringLiteral("/detection");
    QFile blocker(blockerPath);
    QVERIFY(blocker.open(QIODevice::WriteOnly));
    QVERIFY(blocker.write("not a directory") > 0);
    blocker.close();

    cv::Mat image(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    AIPredictionLogService service(&paths);

    QTest::ignoreMessage(QtWarningMsg,
                         QRegularExpression(QStringLiteral("could not create")));
    service.logPrediction(QStringLiteral("detection"), QStringLiteral("frame.png"), image);

    // The blocked path is untouched: nothing was written next to it either.
    QVERIFY(QFileInfo(blockerPath).isFile());
    QCOMPARE(QDir(root.path()).entryList(QDir::NoDotAndDotDot | QDir::AllEntries).size(), 1);
}
