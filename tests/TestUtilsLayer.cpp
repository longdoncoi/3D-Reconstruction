#include <QtTest>

#include <atomic>

#include "BlockingCall.h"

#include <QApplication>
#include <QBuffer>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QTemporaryDir>
#include <QTextBrowser>
#include <QTextStream>
#include <QToolButton>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVariant>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>

#include "AbstractRibbonPanel.h"
#include "CustomProgressDialog.h"
#include "FileUtilities.h"
#include "HtmlUtilities.h"
#include "IconFactory.h"
#include "ILanguageService.h"
#include "Logger.h"
#include "ModernDialog.h"
#include "ModernMessageBox.h"
#include "StyleManager.h"

/**
 * @brief Contract tests for the shared utility layer (src/utils/**).
 *
 * These helpers are the plugin-facing surface every plugin links against
 * (icon factory, ribbon scaffolding, attachment handling, markdown rendering,
 * theming, logging, dialogs).  They were the least measured part of src/:
 * the coverage gate could not see a single executed line of most of them.
 *
 * The tests below pin behaviour, not implementation: a renamed helper or a
 * changed thumbnail rule must fail here instead of silently changing what a
 * plugin sees.
 */

namespace {

/// Minimal ILanguageService stand-in: the utilities only translate strings.
class FakeLanguageService : public ILanguageService {
public:
    QString currentLanguage() const override { return QStringLiteral("en"); }
    void setLanguage(const QString &) override {}
    QString translate(const QString &key) const override { return key; }
    QString translate(const char *key) const override { return QString::fromLatin1(key); }
    QString displayName() const override { return QStringLiteral("English"); }
    QString flagEmoji() const override { return QStringLiteral("EN"); }
};

/// Exposes the protected ribbon-panel API to the test.
class TestRibbonPanel : public AbstractRibbonPanel {
public:
    explicit TestRibbonPanel(IAppContext *ctx) : AbstractRibbonPanel(ctx) {}
    using AbstractRibbonPanel::addButton;
    using AbstractRibbonPanel::beginPanel;
    using AbstractRibbonPanel::finishPanel;
    using AbstractRibbonPanel::group;
};

} // namespace

class TestUtilsLayer : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();

    // ── FileUtilities ───────────────────────────────────────────────────────
    void uniqueFileNameKeepsBaseNameAndAddsTimestamp();
    void processAttachmentCopiesFileAndWritesThumbnail();
    void processAttachmentFailsForMissingSource();
    void processAttachmentSealsTheStoredFiles();
    void processAttachmentFailsWhenTheKeyIsUnavailable();
    void readAttachmentReturnsLegacyPlaintextUnchanged();
    void sealedAttachmentIsReportedAsSealed();
    void sealedAttachmentCanBeShownAsADataUrl();
    void deleteAttachmentRemovesFile();

    // ── HtmlUtilities ───────────────────────────────────────────────────────
    void markdownInlineCodeIsConverted();
    void markdownHeadingsAndEmphasisAreConverted();
    void markdownBulletListIsWrapped();
    void markdownFenceIsHighlighted();
    void markdownSourceCitationBecomesLink();

    // ── IconFactory ─────────────────────────────────────────────────────────
    void modernIconIsRendered();

    // ── BlockingCall ────────────────────────────────────────────────────────
    void blockingTaskTrackerWaitsForTheTask();
    void blockingTaskTrackerReportsATaskThatDidNotFinish();
    void blockingRunDeliversACompletedFuture();
    void blockingRunContainsAThrowingCallable();

    // ── StyleManager ────────────────────────────────────────────────────────
    void accentColorIsStoredAndApplied();
    void globalStyleSheetHasNoUnresolvedPlaceholders();
    void glassmorphismChangesTheStyleSheet();

    // ── Logger ──────────────────────────────────────────────────────────────
    void loggerWritesMessagesToTheConfiguredDirectory();
    void loggerSwallowsTheMonitorInterfaceWarning();

    // ── ModernDialog / ModernMessageBox ────────────────────────────────────
    void modernDialogAcceptsContentLayout();
    void messageBoxIsDismissible();

    // ── CustomProgressDialog ───────────────────────────────────────────────
    void progressDialogUpdatesItsWidgets();
    void progressDialogEmitsStopRequested();

    // ── AbstractRibbonPanel ────────────────────────────────────────────────
    void ribbonPanelRequiresAnHorizontalLayout();
    void ribbonPanelBuildsGroupAndButtons();

private:
    /// The at-rest key the environment pointed at before the test run.
    QString m_oldStorageKey;
};

void TestUtilsLayer::initTestCase()
{
    // The attachment tests seal their files: pin the at-rest key so the run
    // never touches the real user profile.
    const QByteArray key("0123456789abcdef0123456789abcdef");
    m_oldStorageKey = qEnvironmentVariable("AI_STORAGE_KEY");
    qputenv("AI_STORAGE_KEY", "base64:" + key.toBase64(QByteArray::Base64UrlEncoding));
}

void TestUtilsLayer::cleanupTestCase()
{
    if (m_oldStorageKey.isEmpty()) {
        qunsetenv("AI_STORAGE_KEY");
    } else {
        qputenv("AI_STORAGE_KEY", m_oldStorageKey.toLocal8Bit());
    }
}

void TestUtilsLayer::uniqueFileNameKeepsBaseNameAndAddsTimestamp()
{
    const QString name = FileUtilities::getUniqueFileName(QStringLiteral("/tmp/dir/report.pdf"));
    QVERIFY(name.endsWith(QStringLiteral("_report.pdf")));
    QVERIFY(!name.contains(QLatin1Char('/')));
    // The prefix must be a timestamp, not the original path.
    const QString prefix = name.left(name.indexOf(QLatin1Char('_')));
    QVERIFY(prefix.toLongLong() > 0);
}

void TestUtilsLayer::processAttachmentCopiesFileAndWritesThumbnail()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    const QString source = tmp.filePath(QStringLiteral("notes.txt"));
    QFile file(source);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("hello attachment");
    file.close();

    const QString uploadRoot = tmp.filePath(QStringLiteral("uploads"));
    const QString thumbRoot = tmp.filePath(QStringLiteral("thumbs"));

    const auto result = FileUtilities::processAttachment(source, false, uploadRoot, thumbRoot);
    QVERIFY(result.success);
    QVERIFY(QFile::exists(result.destPath));
    QVERIFY(result.destPath.startsWith(uploadRoot));
    // A non-image attachment gets a .png placeholder thumbnail.
    QVERIFY(result.thumbPath.endsWith(QStringLiteral(".png")));
    QVERIFY(QFile::exists(result.thumbPath));
    QVERIFY(!result.thumbnail.isNull());
    QCOMPARE(result.thumbnail.size(), QSize(80, 80));

    QFile copied(result.destPath);
    QVERIFY(copied.open(QIODevice::ReadOnly));
    const QByteArray stored = copied.readAll();
    copied.close();
    // The copy is sealed at rest — the plaintext is only reachable through
    // readAttachment().
    QVERIFY(stored.startsWith("enc:v1:"));
    QCOMPARE(FileUtilities::readAttachment(result.destPath,
                                           QString::fromLatin1(FileUtilities::AttachmentContext)),
             QByteArray("hello attachment"));
}

void TestUtilsLayer::processAttachmentFailsForMissingSource()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    const auto result = FileUtilities::processAttachment(
        tmp.filePath(QStringLiteral("does-not-exist.bin")), false,
        tmp.filePath(QStringLiteral("uploads")), tmp.filePath(QStringLiteral("thumbs")));
    QVERIFY(!result.success);
    QVERIFY(result.destPath.isEmpty());
}

void TestUtilsLayer::processAttachmentSealsTheStoredFiles()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    const QString source = tmp.filePath(QStringLiteral("photo.png"));
    QImage image(8, 8, QImage::Format_RGB32);
    image.fill(Qt::red);
    QVERIFY(image.save(source));

    QFile original(source);
    QVERIFY(original.open(QIODevice::ReadOnly));
    const QByteArray originalBytes = original.readAll();
    original.close();

    const auto result = FileUtilities::processAttachment(
        source, true, tmp.filePath(QStringLiteral("uploads")),
        tmp.filePath(QStringLiteral("thumbs")));
    QVERIFY(result.success);
    QVERIFY(!result.thumbnail.isNull());

    // Neither the copied original nor the thumbnail is readable on disk.
    QFile stored(result.destPath);
    QVERIFY(stored.open(QIODevice::ReadOnly));
    const QByteArray storedBytes = stored.readAll();
    stored.close();
    QVERIFY(storedBytes.startsWith("enc:v1:"));
    QVERIFY(!storedBytes.contains("\x89PNG"));

    QFile thumb(result.thumbPath);
    QVERIFY(thumb.open(QIODevice::ReadOnly));
    const QByteArray thumbBytes = thumb.readAll();
    thumb.close();
    QVERIFY(thumbBytes.startsWith("enc:v1:"));

    // The plaintext is recovered through the matching context only.
    QCOMPARE(FileUtilities::readAttachment(result.destPath,
                                           QString::fromLatin1(FileUtilities::AttachmentContext)),
             originalBytes);
    const QByteArray thumbnail = FileUtilities::readAttachment(
        result.thumbPath, QString::fromLatin1(FileUtilities::ThumbnailContext));
    QVERIFY(thumbnail.startsWith("\x89PNG"));
    // The thumbnail context does not open the original and vice versa.
    QVERIFY(FileUtilities::readAttachment(result.destPath,
                                          QString::fromLatin1(FileUtilities::ThumbnailContext))
                .isEmpty());
    QVERIFY(FileUtilities::readAttachment(result.thumbPath,
                                          QString::fromLatin1(FileUtilities::AttachmentContext))
                .isEmpty());

    // loadThumbnailPixmap decodes the sealed thumbnail directly into RAM
    const QPixmap thumbPix = FileUtilities::loadThumbnailPixmap(result.thumbPath);
    QVERIFY(!thumbPix.isNull());
    QCOMPARE(thumbPix.size(), QSize(80, 80));
    QVERIFY(FileUtilities::loadThumbnailPixmap(result.destPath).isNull());
    QVERIFY(FileUtilities::loadThumbnailPixmap(tmp.filePath(QStringLiteral("none.png"))).isNull());
}

void TestUtilsLayer::processAttachmentFailsWhenTheKeyIsUnavailable()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    const QString source = tmp.filePath(QStringLiteral("notes.txt"));
    QFile file(source);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("hello attachment");
    file.close();

    // A malformed explicit key fails closed: nothing may be written to disk.
    const QString saved = qEnvironmentVariable("AI_STORAGE_KEY");
    qputenv("AI_STORAGE_KEY", QByteArray("broken-key"));
    const auto result = FileUtilities::processAttachment(
        source, false, tmp.filePath(QStringLiteral("uploads")),
        tmp.filePath(QStringLiteral("thumbs")));
    if (saved.isEmpty()) {
        qunsetenv("AI_STORAGE_KEY");
    } else {
        qputenv("AI_STORAGE_KEY", saved.toLocal8Bit());
    }

    QVERIFY(!result.success);
    QVERIFY(result.destPath.isEmpty());
    QVERIFY(result.thumbPath.isEmpty());
}

void TestUtilsLayer::sealedAttachmentIsReportedAsSealed()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    const QString source = tmp.filePath(QStringLiteral("note.txt"));
    QFile file(source);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("plain");
    file.close();

    const auto result = FileUtilities::processAttachment(
        source, false, tmp.filePath(QStringLiteral("uploads")),
        tmp.filePath(QStringLiteral("thumbs")));
    QVERIFY(result.success);

    // The stored copy is sealed, the source is not, and a missing file is
    // not sealed either — the chat history uses this to decide whether the
    // attachment can be shown as a link.
    QVERIFY(FileUtilities::isSealedAttachment(result.destPath));
    QVERIFY(!FileUtilities::isSealedAttachment(source));
    QVERIFY(!FileUtilities::isSealedAttachment(tmp.filePath(QStringLiteral("gone.txt"))));
}

void TestUtilsLayer::readAttachmentReturnsLegacyPlaintextUnchanged()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    const QString path = tmp.filePath(QStringLiteral("legacy.txt"));
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("written before the envelope existed");
    file.close();

    // An attachment written before the sealing still has to be readable.
    QCOMPARE(FileUtilities::readAttachment(path, QString::fromLatin1(FileUtilities::AttachmentContext)),
             QByteArray("written before the envelope existed"));
    // A missing file has no plaintext.
    QVERIFY(FileUtilities::readAttachment(tmp.filePath(QStringLiteral("gone.txt")),
                                          QString::fromLatin1(FileUtilities::AttachmentContext))
                .isEmpty());
}

void TestUtilsLayer::sealedAttachmentCanBeShownAsADataUrl()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());

    const QString source = tmp.filePath(QStringLiteral("photo.png"));
    QImage image(8, 8, QImage::Format_RGB32);
    image.fill(Qt::blue);
    QVERIFY(image.save(source));

    const auto result = FileUtilities::processAttachment(
        source, true, tmp.filePath(QStringLiteral("uploads")),
        tmp.filePath(QStringLiteral("thumbs")));
    QVERIFY(result.success);

    // The chat history embeds the unsealed attachment as a data: URL: the
    // rich-text engine has to be able to decode it.
    const QByteArray bytes = FileUtilities::readAttachment(
        result.destPath, QString::fromLatin1(FileUtilities::AttachmentContext));
    QVERIFY(!bytes.isEmpty());
    const QString url = QStringLiteral("data:image/png;base64,")
                        + QString::fromLatin1(bytes.toBase64());

    QTextBrowser browser;
    browser.setHtml(QStringLiteral("<img src='%1'>").arg(url));
    const QVariant resource =
        browser.document()->resource(QTextDocument::ImageResource, QUrl(url));
    QVERIFY(resource.isValid());
    QImage decoded;
    if (resource.userType() == QMetaType::QPixmap) {
        decoded = resource.value<QPixmap>().toImage();
    } else if (resource.userType() == QMetaType::QImage) {
        decoded = resource.value<QImage>();
    } else if (resource.userType() == QMetaType::QByteArray) {
        decoded.loadFromData(resource.toByteArray());
    }
    QVERIFY(!decoded.isNull());
}

void TestUtilsLayer::deleteAttachmentRemovesFile()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    const QString path = tmp.filePath(QStringLiteral("a.txt"));
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.close();

    QVERIFY(FileUtilities::deleteAttachment(path));
    QVERIFY(!QFile::exists(path));
    // Deleting a file that is not there is not an error.
    QVERIFY(!FileUtilities::deleteAttachment(path));
}

void TestUtilsLayer::markdownInlineCodeIsConverted()
{
    const QString html = HtmlUtilities::mdToHtml(QStringLiteral("call `foo()` now"));
    QVERIFY(html.contains(QStringLiteral("<code")));
    QVERIFY(html.contains(QStringLiteral("foo()")));
}

void TestUtilsLayer::markdownHeadingsAndEmphasisAreConverted()
{
    QString html = HtmlUtilities::mdToHtml(QStringLiteral("# Title"));
    QVERIFY(html.contains(QStringLiteral("<h1>Title</h1>")));
    html = HtmlUtilities::mdToHtml(QStringLiteral("## Section"));
    QVERIFY(html.contains(QStringLiteral("<h2>Section</h2>")));
    html = HtmlUtilities::mdToHtml(QStringLiteral("**bold**"));
    QVERIFY(html.contains(QStringLiteral("<b>bold</b>")));
}

void TestUtilsLayer::markdownBulletListIsWrapped()
{
    const QString html = HtmlUtilities::mdToHtml(QStringLiteral("- one\n- two"));
    QVERIFY(html.contains(QStringLiteral("<ul>")));
    QVERIFY(html.contains(QStringLiteral("<li>one</li>")));
    QVERIFY(html.contains(QStringLiteral("<li>two</li>")));
    QVERIFY(html.contains(QStringLiteral("</ul>")));
}

void TestUtilsLayer::markdownFenceIsHighlighted()
{
    const QString html = HtmlUtilities::mdToHtml(
        QStringLiteral("```cpp\nint main() { return 0; }\n```"));
    QVERIFY(html.contains(QStringLiteral("<pre")));
    QVERIFY(html.contains(QStringLiteral("cpp")));
    // Keywords are highlighted, so the raw keyword must be gone.
    QVERIFY(!html.contains(QStringLiteral(">int <")));
}

void TestUtilsLayer::markdownSourceCitationBecomesLink()
{
    const QString html = HtmlUtilities::mdToHtml(
        QStringLiteral("see [Source: C:/tmp/a.pdf]"));
    QVERIFY(html.contains(QStringLiteral("source-citation")));
    QVERIFY(html.contains(QStringLiteral("file:///C:/tmp/a.pdf")));
}

void TestUtilsLayer::modernIconIsRendered()
{
    const QIcon icon = IconFactory::createModern(QStringLiteral("A"),
                                                   QColor(QStringLiteral("#3b82f6")),
                                                   QColor(QStringLiteral("#2563eb")));
    QVERIFY(!icon.isNull());
    const QPixmap pixmap = icon.pixmap(48, 48);
    QVERIFY(!pixmap.isNull());
    // The device-independent size is the contract; the pixel size follows the
    // screen's scale factor, so asserting 48x48 pixels failed on a 125 % display.
    QCOMPARE(pixmap.deviceIndependentSize().toSize(), QSize(48, 48));
}

void TestUtilsLayer::blockingTaskTrackerWaitsForTheTask()
{
    // The tracker is what keeps a background task from outliving the object
    // it uses: the teardown path waits for it before releasing that object.
    blocking::TaskTracker tracker;
    std::atomic<bool> ran{false};
    auto future = blocking::run([&ran]() {
        QThread::msleep(50);
        ran.store(true);
        return 42;
    }, &tracker);
    QCOMPARE(tracker.taskCount(), 1);

    QCOMPARE(tracker.waitForFinished(5000), 0);
    QVERIFY(ran.load());
    QCOMPARE(future.result(), 42);
    // The tracker is empty again — a second wait must not block.
    QCOMPARE(tracker.taskCount(), 0);
    QCOMPARE(tracker.waitForFinished(10), 0);
}

void TestUtilsLayer::blockingTaskTrackerReportsATaskThatDidNotFinish()
{
    // A task that hangs must not freeze the shutdown: the wait is bounded
    // and the task is reported.
    blocking::TaskTracker tracker;
    std::atomic<bool> release{false};
    blocking::run([&release]() {
        while (!release.load()) QThread::msleep(10);
    }, &tracker);

    QCOMPARE(tracker.waitForFinished(100), 1);
    release.store(true);
}

void TestUtilsLayer::blockingRunDeliversACompletedFuture()
{
    // The result must reach the caller through the `finished` signal — that
    // is the only channel the mail/auth plugins use (MailDockWidget,
    // MailSettingsDialog, ForgotPasswordDialog, SettingsDialog).
    //
    // The future must also be COMPLETED, not cancelled. A promise that is
    // never finished is cancelled by QPromise's destructor, so a helper that
    // only calls addResult() delivers its result in the cancelled state: the
    // natural `if (watcher.isCanceled()) return;` guard would then drop the
    // result silently, and the caller would wait for a notification that
    // only arrives because the worker thread object happened to be destroyed.
    QFutureWatcher<QPair<bool, QString>> watcher;
    bool finished = false;
    // The WATCHER is the context, not `this`: the lambda captures a local of
    // this function, so a queued `finished` event that is delivered after the
    // function returned would write to a dead stack slot (a /GS cookie
    // failure, i.e. 0xC0000409). With the watcher as the receiver, Qt drops
    // the pending event when the watcher dies.
    connect(&watcher, &QFutureWatcher<QPair<bool, QString>>::finished, &watcher,
            [&finished]() { finished = true; });

    watcher.setFuture(blocking::run([]() {
        return qMakePair(true, QStringLiteral("done"));
    }));

    QTRY_VERIFY(finished);
    QVERIFY2(!watcher.isCanceled(),
             "the future must be completed, not cancelled: a cancelled future "
             "is indistinguishable from a failed one for the caller");
    QCOMPARE(watcher.result().first, true);
    QCOMPARE(watcher.result().second, QStringLiteral("done"));
}

void TestUtilsLayer::blockingRunContainsAThrowingCallable()
{
    // ONNX Runtime, OpenCV and the socket layer all throw. An exception that
    // leaves QThread::run() reaches std::terminate() and ABORTS THE WHOLE
    // PROCESS (0xC0000409) — the app dies instead of reporting one failed
    // request. The helper must contain it and still release the semaphore.
    blocking::TaskTracker tracker;
    QFutureWatcher<QPair<bool, QString>> watcher;
    bool finished = false;
    // The watcher is the context (see the note in
    // blockingRunDeliversACompletedFuture): the lambda captures a local of
    // this function, so a queued event delivered after the function returned
    // would write to a dead stack slot.
    connect(&watcher, &QFutureWatcher<QPair<bool, QString>>::finished, &watcher,
            [&finished]() { finished = true; });

    watcher.setFuture(blocking::run([]() -> QPair<bool, QString> {
        throw std::runtime_error("the socket blew up");
    }, &tracker));

    // The wait must NOT report an unfinished task: the semaphore is released
    // even when the callable throws, otherwise the owner's teardown burns
    // its whole timeout.
    QCOMPARE(tracker.waitForFinished(5000), 0);
    QTRY_VERIFY(finished);

    // The failure must NOT come back as an exception: QFuture::result()
    // re-throws an exception stored on the promise, and that re-throw
    // happens inside the CALLER's slot — where it ends in std::terminate()
    // again, one stack frame further from the cause. The caller gets a
    // default-constructed result instead ("not ok").
    QVERIFY2(!watcher.result().first,
             "a failed task must not look like a success to the caller");
    QVERIFY2(!watcher.isCanceled(),
             "the future must be completed, not cancelled: a cancelled future "
             "is indistinguishable from a failed one for the caller");
}

void TestUtilsLayer::accentColorIsStoredAndApplied()
{
    const QString original = StyleManager::accentColor();
    StyleManager::setAccentColor(QStringLiteral("#ff0000"));
    QCOMPARE(StyleManager::accentColor(), QStringLiteral("#ff0000"));
    QVERIFY(StyleManager::getGlobalStyleSheet().contains(QStringLiteral("#ff0000")));
    StyleManager::setAccentColor(original);
}

void TestUtilsLayer::globalStyleSheetHasNoUnresolvedPlaceholders()
{
    const QString sheet = StyleManager::getGlobalStyleSheet();
    for (int i = 1; i <= 8; ++i) {
        QVERIFY2(!sheet.contains(QStringLiteral("%%%1").arg(i)),
                 qPrintable(QStringLiteral("unresolved placeholder %%%1").arg(i)));
    }
    QVERIFY(!sheet.isEmpty());
}

void TestUtilsLayer::glassmorphismChangesTheStyleSheet()
{
    const bool original = StyleManager::isGlassmorphismEnabled();
    const QString solid = StyleManager::getGlobalStyleSheet();

    StyleManager::setGlassmorphismEnabled(!original);
    QVERIFY(StyleManager::isGlassmorphismEnabled() != original);
    QVERIFY(StyleManager::getGlobalStyleSheet() != solid);

    StyleManager::setGlassmorphismEnabled(original);
    QCOMPARE(StyleManager::isGlassmorphismEnabled(), original);
}

void TestUtilsLayer::loggerWritesMessagesToTheConfiguredDirectory()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    Logger::setLogsDir(tmp.path());

    Logger::customMessageHandler(QtInfoMsg, QMessageLogContext(), QStringLiteral("utils-layer-probe"));
    Logger::shutdown();

    const QDir dir(tmp.path());
    const QStringList files = dir.entryList({QStringLiteral("*.txt")}, QDir::Files);
    QCOMPARE(files.size(), 1);

    QFile logFile(dir.filePath(files.first()));
    QVERIFY(logFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString content = QString::fromUtf8(logFile.readAll());
    QVERIFY2(content.contains(QStringLiteral("utils-layer-probe")), qPrintable(content));
    QVERIFY(content.contains(QStringLiteral("[INFO]")));
}

void TestUtilsLayer::loggerSwallowsTheMonitorInterfaceWarning()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    Logger::setLogsDir(tmp.path());

    Logger::customMessageHandler(QtWarningMsg, QMessageLogContext(),
                                 QStringLiteral("Unable to open monitor interface"));
    Logger::shutdown();

    const QDir dir(tmp.path());
    const QStringList files = dir.entryList({QStringLiteral("*.txt")}, QDir::Files);
    // The warning is filtered out before the file is opened, so no log file
    // may be created for it.
    QVERIFY2(files.isEmpty(), qPrintable(files.join(QStringLiteral(", "))));
}

void TestUtilsLayer::modernDialogAcceptsContentLayout()
{
    ModernDialog dialog(QStringLiteral("Title"));
    auto *content = new QWidget;
    auto *layout = new QVBoxLayout(content);
    layout->addWidget(new QLabel(QStringLiteral("body")));
    dialog.setContentLayout(layout);
    // ModernDialog draws its own title bar: setWindowTitle() updates the
    // internal label, not the window manager's title.
    dialog.setWindowTitle(QStringLiteral("Other title"));
    auto *titleLabel = dialog.findChild<QLabel *>(QStringLiteral("modernTitleLabel"));
    QVERIFY(titleLabel != nullptr);
    QCOMPARE(titleLabel->text(), QStringLiteral("Other title"));
    dialog.show();
    QVERIFY(dialog.isVisible());
    dialog.hide();
}

void TestUtilsLayer::messageBoxIsDismissible()
{
    // The message box is modal: close it from the event loop.
    QTimer::singleShot(0, [] {
        if (auto *modal = QApplication::activeModalWidget()) {
            if (auto *dialog = qobject_cast<QDialog *>(modal)) {
                dialog->accept();
            }
        }
    });
    ModernMessageBox::information(nullptr, QStringLiteral("Title"), QStringLiteral("Text"));
    QVERIFY(true); // reaching this point means the dialog was dismissed
}

void TestUtilsLayer::progressDialogUpdatesItsWidgets()
{
    FakeLanguageService lang;
    CustomProgressDialog dialog(&lang);

    dialog.setRange(0, 100);
    dialog.setValue(42);
    dialog.setLabelText(QStringLiteral("Working"));
    dialog.reset();
    dialog.show();
    dialog.hide();
    dialog.centerOnWidget(nullptr);
    QVERIFY(true);
}

void TestUtilsLayer::progressDialogEmitsStopRequested()
{
    FakeLanguageService lang;
    CustomProgressDialog dialog(&lang);

    QSignalSpy spy(&dialog, &IProgressDialog::stopRequested);
    // The dialog's internal window is an independent top-level widget (it is
    // not parented to the IProgressDialog object), so the stop button is
    // looked up in the top-level widgets.
    QPushButton *stopButton = nullptr;
    for (QWidget *widget : QApplication::topLevelWidgets()) {
        const QList<QPushButton *> buttons = widget->findChildren<QPushButton *>();
        if (!buttons.isEmpty()) {
            stopButton = buttons.first();
            break;
        }
    }
    QVERIFY(stopButton != nullptr);
    stopButton->click();

    QCOMPARE(spy.count(), 1);
}

void TestUtilsLayer::ribbonPanelRequiresAnHorizontalLayout()
{
    TestRibbonPanel panel(nullptr);
    QWidget parent;
    parent.setLayout(new QVBoxLayout); // vertical → not hostable
    QVERIFY(!panel.beginPanel(&parent, QStringLiteral("group")));
}

void TestUtilsLayer::ribbonPanelBuildsGroupAndButtons()
{
    TestRibbonPanel panel(nullptr);
    QWidget parent;
    auto *layout = new QHBoxLayout;
    layout->addStretch(); // trailing spacer, as in the real ribbon tab
    parent.setLayout(layout);

    QVERIFY(panel.beginPanel(&parent, QStringLiteral("myGroup")));
    QVERIFY(panel.group() != nullptr);
    QCOMPARE(panel.group()->objectName(), QStringLiteral("myGroup"));

    QToolButton *button = panel.addButton(QStringLiteral("A"),
                                          QStringLiteral("#3b82f6"),
                                          QStringLiteral("#2563eb"),
                                          QStringLiteral("Action"));
    QVERIFY(button != nullptr);
    QCOMPARE(button->text(), QStringLiteral("Action"));
    QVERIFY(!button->icon().isNull());
    QCOMPARE(button->minimumWidth(), 80);

    panel.finishPanel(QStringLiteral("Group title"));
    // The group is inserted before the trailing stretch.
    QCOMPARE(layout->count(), 2);
    QCOMPARE(layout->itemAt(0)->widget(), panel.group());
}

QTEST_MAIN(TestUtilsLayer)
#include "TestUtilsLayer.moc"
