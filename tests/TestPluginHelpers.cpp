#include <QtTest>

#include <QDir>
#include <QElapsedTimer>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QTemporaryDir>
#include <QWidget>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "AIPredictionLogger.h"
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

    // ── AIPredictionLogger ──────────────────────────────────────────────────
    void predictionLoggerIgnoresANullPathProvider();
    void predictionLoggerWritesTheResultImage();
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
    AIPredictionLogger::logAsync(nullptr, "detection", "frame.png", image);
    QVERIFY(true);
}

void TestPluginHelpers::predictionLoggerWritesTheResultImage()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    StubAppPaths paths(root.path());

    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    AIPredictionLogger::logAsync(&paths, "detection", "frame.png", image);

    // The write happens on the global thread pool: wait for the file.
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

QTEST_MAIN(TestPluginHelpers)
#include "TestPluginHelpers.moc"
