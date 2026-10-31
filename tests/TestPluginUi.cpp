#include <QtTest>

#include <QDialog>
#include <QDateTime>
#include <QFile>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QListWidget>
#include <QPointer>
#include <QPushButton>
#include <QTabWidget>
#include <QTextBrowser>
#include <QTimer>
#include <QToolButton>
#include <QWidget>

#include <opencv2/imgcodecs.hpp>

#include "core/AtRestCipher.h"

#include "AIAssistantRibbonUI.h"
#include "AIProcessorPlugin.h"
#include "AIProcessorRibbonUI.h"
#include "AITrainDockWidget.h"
#include "ChatMessageRenderer.h"
#include "FileUtilities.h"
#include "IMailService.h"
#include "MailDockWidget.h"
#include "MailFilterDialog.h"
#include "MailSettingsDialog.h"
#include "SendMailRibbonUI.h"
#include "ViewerRibbonUI.h"
#include "ViewerListUI.h"
#include "ViewerViewModel.h"
#include "mock/PluginContextMock.h"

/**
 * @brief The remaining UI surface of the plugins: the ribbon panels, the mail
 * dialogs, the training dock and the viewer view model.
 *
 * The ribbon panels share one recipe (AbstractRibbonPanel): a group box with
 * an icon-painted row of tool buttons, installed into a tab panel.  The mail
 * dialogs are modal — a guard closes them so the test cannot hang.
 */
class TestPluginUi : public QObject {
    Q_OBJECT

private slots:
    void sendMailRibbonIsBuilt();
    void aiAssistantRibbonIsBuilt();
    void aiProcessorRibbonIsBuilt();
    void viewerRibbonIsBuilt();
    void mailFilterDialogCanBeDismissed();
    void mailFilterDialogCanRemoveKeywordsAndAccept();
    void mailSettingsDialogCanBeDismissed();
    void mailDockShowsTheInboxAndRetranslates();
    void viewerViewModelIgnoresEmptyPaths();
    void viewerViewModelLoadsImagesAndDicom();
    void viewerListUiFollowsTheImageList();
    void trainingDockIsBuiltAndRetranslated();
    void detectionInferenceRunsOffTheGuiThread();
    void segmentationInferenceRunsOffTheGuiThread();
    void oneInferenceAtATime();
    void inferenceStaleResultDroppedOnImageChange();
    void actionButtonsRemainEnabledDuringNavigation();
    void aThrowingInferenceIsReportedOnce();
    void anEmptyInferenceResultIsReportedOnce();
    void aTrackingFailureIsNotReportedAsSuccess();
    void chatHistoryRendersTheMessages();
    void theAgentTraceRendersItsSteps();
    void theAgentTraceRendersEveryStepKind();
    void userAttachmentsAreRenderedSafely();
    void theThinkingIndicatorFollowsTheInsertIndex();
};

namespace {

/// A tab panel like the shell provides it: a horizontal layout to host a group.
QWidget *makePanel()
{
    auto *panel = new QWidget;
    panel->setLayout(new QHBoxLayout);
    return panel;
}

/// Closes a modal dialog as soon as it appears.
class ModalDialogGuard : public QObject {
public:
    ModalDialogGuard()
    {
        m_timer.setInterval(50);
        connect(&m_timer, &QTimer::timeout, this, [this] {
            if (auto *modal = QApplication::activeModalWidget()) {
                if (modal != m_last) {
                    m_last = modal;
                    ++m_seen;
                }
                if (auto *dialog = qobject_cast<QDialog *>(modal)) {
                    dialog->reject();
                } else {
                    modal->close();
                }
            }
        });
    }

    void start() { m_timer.start(); }
    void stop() { m_timer.stop(); }

    /// How many DISTINCT modal dialogs the guard has seen. A message box runs
    /// `exec()`, so it stays up until this timer dismisses it — every box that
    /// was actually shown is counted exactly once. That turns "does the UI
    /// report this as a success?" into an assertion instead of a guess.
    int seenDialogs() const { return m_seen; }

private:
    QTimer m_timer;
    QPointer<QWidget> m_last;
    int m_seen = 0;
};

/// A small image on disk: the inference task reads the path the viewer reports
/// (cv::imread runs on the worker thread), so the file has to exist.
QString writeTestImage(const QTemporaryDir &dir, const QString &name = QStringLiteral("frame.png"))
{
    const QString path = dir.filePath(name);
    const cv::Mat image(16, 16, CV_8UC3, cv::Scalar(10, 20, 30));
    if (!cv::imwrite(path.toStdString(), image)) return {};
    return path;
}

/// Initializes the plugin and guarantees its bounded teardown on every exit
/// path.  A failed assertion must not leave an inference task calling into the
/// mock services after the test function's stack has been unwound.
struct RunningAIProcessorPlugin {
    explicit RunningAIProcessorPlugin(PluginMock::AppContext &context)
        : initialized(plugin.initialize(&context)) {}
    ~RunningAIProcessorPlugin()
    {
        if (initialized) plugin.cleanup();
    }

    AIProcessorPlugin plugin;
    bool initialized = false;
};

} // namespace

void TestPluginUi::sendMailRibbonIsBuilt()
{
    PluginMock::AppContext context(QDir::tempPath());
    QWidget panel;
    panel.setLayout(new QHBoxLayout);

    SendMailRibbonUI ribbon(&context, &panel);
    QVERIFY(ribbon.btnToggleMail() != nullptr);
    QVERIFY(ribbon.btnSettings() != nullptr);
    QVERIFY(ribbon.groupMail() != nullptr);
}

void TestPluginUi::aiAssistantRibbonIsBuilt()
{
    PluginMock::AppContext context(QDir::tempPath());
    QWidget panel;
    panel.setLayout(new QHBoxLayout);

    AIAssistantRibbonUI ribbon(&context, &panel);
    QVERIFY(ribbon.btnToggleAssistant() != nullptr);
}

void TestPluginUi::aiProcessorRibbonIsBuilt()
{
    PluginMock::AppContext context(QDir::tempPath());
    QWidget panel;
    panel.setLayout(new QHBoxLayout);

    AIProcessorRibbonUI ribbon(&context, &panel);
    QVERIFY(ribbon.btnDet() != nullptr);
}

void TestPluginUi::viewerRibbonIsBuilt()
{
    PluginMock::AppContext context(QDir::tempPath());
    QWidget panel;
    panel.setLayout(new QHBoxLayout);

    ViewerRibbonUI ribbon(&context, &panel);
    QVERIFY(ribbon.btnLoad2D() != nullptr);
}

void TestPluginUi::mailFilterDialogCanBeDismissed()
{
    PluginMock::AppContext context(QDir::tempPath());
    QStringList keywords{QStringLiteral("keep")};

    ModalDialogGuard guard;
    guard.start();
    // The dialog is modal: the guard dismisses it, so edit() returns false.
    const bool accepted = MailFilterDialog::edit(nullptr, &context, keywords);
    guard.stop();

    QVERIFY(!accepted);
    // The keywords must survive the cancelled edit untouched.
    QCOMPARE(keywords, QStringList{QStringLiteral("keep")});
}

void TestPluginUi::mailFilterDialogCanRemoveKeywordsAndAccept()
{
    PluginMock::AppContext context(QDir::tempPath());
    QStringList keywords{QStringLiteral("remove_me"), QStringLiteral("keep_me")};

    QTimer::singleShot(50, [&]() {
        QWidget *modal = QApplication::activeModalWidget();
        if (auto *dlg = qobject_cast<QDialog *>(modal)) {
            auto *list = dlg->findChild<QListWidget *>();
            if (list && list->count() > 0) {
                list->setCurrentRow(0);
                const auto buttons = dlg->findChildren<QPushButton *>();
                for (auto *btn : buttons) {
                    if (btn->text() == context.translate("mail.remove_filter")) {
                        btn->click();
                        break;
                    }
                }
            }
            dlg->accept();
        }
    });

    const bool accepted = MailFilterDialog::edit(nullptr, &context, keywords);
    QVERIFY(accepted);
    QCOMPARE(keywords, QStringList{QStringLiteral("keep_me")});
}

void TestPluginUi::mailSettingsDialogCanBeDismissed()
{
    PluginMock::AppContext context(QDir::tempPath());
    PluginMock::MailService mail;

    ModalDialogGuard guard;
    guard.start();
    MailSettingsDialog::show(nullptr, &context, &mail);
    guard.stop();

    QVERIFY(true); // reaching this point means the modal dialog was dismissed
}

void TestPluginUi::mailDockShowsTheInboxAndRetranslates()
{
    // The mail dock is the only UI that renders the inbox: it owns the fetch
    // (on a worker thread — the mail stack is blocking), the list and the
    // preview. A defect here is invisible to the compiler: the dock still
    // builds, it just never fills or never retranslates.
    PluginMock::AppContext context(QDir::tempPath());

    MailMessage message;
    message.uid = QStringLiteral("42");
    message.from = QStringLiteral("alice@example.com");
    message.to = QStringLiteral("bot@example.com");
    message.subject = QStringLiteral("Quarterly report");
    message.date = QDateTime(QDate(2026, 1, 2), QTime(3, 4));
    message.htmlBody = QStringLiteral("<p>the numbers are in</p>");
    context.mailService().inbox = {message};

    MailDockWidget dock(&context);
    QVERIFY(dock.dockWidget() != nullptr);

    // A language change re-labels the dock itself, not only its pages: the
    // dock is created once and survives every language switch.
    dock.retranslate();
    QCOMPARE(dock.dockWidget()->windowTitle(), context.translate("mail.dock_title"));

    auto *tabs = dock.dockWidget()->findChild<QTabWidget *>(QStringLiteral("mailTabs"));
    QVERIFY(tabs != nullptr);
    dock.showCompose();
    QCOMPARE(tabs->currentIndex(), 1);
    dock.showInbox();
    QCOMPARE(tabs->currentIndex(), 0);

    // The fetch is a blocking::run task, so the list fills asynchronously.
    auto *inbox = tabs->widget(0)->findChild<QListWidget *>();
    QVERIFY(inbox != nullptr);
    QTRY_VERIFY_WITH_TIMEOUT(inbox->count() == 1, 10000);
    QCOMPARE(context.mailService().inboxCalls, 1);

    auto *preview = tabs->widget(0)->findChild<QTextBrowser *>();
    QVERIFY(preview != nullptr);
    inbox->setCurrentRow(0);
    QVERIFY2(preview->toPlainText().contains(QStringLiteral("Quarterly report")),
             qPrintable(preview->toPlainText()));

    // A second visit reuses the fetched list: showing the inbox again must not
    // hit the server (the refresh button is the explicit way to do that).
    dock.showInbox();
    QCOMPARE(context.mailService().inboxCalls, 1);
}

void TestPluginUi::viewerViewModelIgnoresEmptyPaths()
{
    PluginMock::AppContext context(QDir::tempPath());
    ViewerViewModel model(&context);

    // An empty path is a no-op — no scene, no viewer, no progress signals.
    model.load2DImage(QString());
    model.load3DModel(QString());
    QVERIFY(true);
}

void TestPluginUi::viewerViewModelLoadsImagesAndDicom()
{
    PluginMock::AppContext context(QDir::tempPath());
    ViewerViewModel model(&context);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString first = dir.filePath(QStringLiteral("first.png"));
    const QString second = dir.filePath(QStringLiteral("second.jpg"));
    const QString model3d = dir.filePath(QStringLiteral("model.obj"));
    for (const QString &path : {first, second, model3d}) {
        QFile file(path);
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.write("not-a-real-image");
    }

    QSignalSpy started(&model, &ViewerViewModel::loadingStarted);
    QSignalSpy finished(&model, &ViewerViewModel::loadingFinished);
    QSignalSpy navigation(&model, &ViewerViewModel::showNavigationUI);

    // A load that returns an actor takes the texture path: the scene gets the
    // actor, the camera is reset and the folder's images are published in
    // name order with the loaded one selected. (The mock scene returns no
    // actor by default, which is the "reader failed" path.)
    context.sceneService().imageActor = vtkSmartPointer<vtkActor>::New();
    model.load2DImage(second);

    QCOMPARE(started.count(), 1);
    QCOMPARE(finished.count(), 1);
    QCOMPARE(navigation.count(), 1);
    QCOMPARE(navigation.first().at(0).toBool(), true);
    QCOMPARE(context.viewerService().getCurrent2DImagePath(), second);
    QCOMPARE(context.viewerService().imageList(),
             (QStringList{QStringLiteral("first.png"), QStringLiteral("second.jpg")}));
    QCOMPARE(context.viewerService().imageIndex(), 1);

    // A 3D load clears the 2D state and hands the file to the scene loader.
    model.load3DModel(model3d);
    QCOMPARE(started.count(), 2);
    QCOMPARE(finished.count(), 2);
    QCOMPARE(navigation.last().at(0).toBool(), false);
    QVERIFY(context.viewerService().getCurrent2DImagePath().isEmpty());
    QVERIFY(context.viewerService().imageList().isEmpty());
    QCOMPARE(context.sceneService().lastObjPath, model3d);

    // A DICOM directory is delegated to the scene pipeline. The mock reports
    // success: the failure path shows a modal box, which the test cannot
    // dismiss without an event loop (and a modal box is not the subject).
    context.sceneService().dicomSucceeds = true;
    model.loadDicom(dir.path());
    QCOMPARE(context.sceneService().dicomCalls, 1);
    QCOMPARE(started.count(), 3);
    QCOMPARE(finished.count(), 3);
    QCOMPARE(navigation.last().at(0).toBool(), false);
}

void TestPluginUi::viewerListUiFollowsTheImageList()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString first = dir.filePath(QStringLiteral("a.png"));
    const QString second = dir.filePath(QStringLiteral("b.png"));
    for (const QString &path : {first, second}) {
        QFile file(path);
        QVERIFY(file.open(QIODevice::WriteOnly));
        file.write("not-a-real-image");
    }

    ViewerListUI list(&context);
    list.setupUI();
    QListWidget *widget = list.widget();
    QVERIFY(widget != nullptr);
    // Hidden until the viewer publishes a list.
    QVERIFY(widget->isHidden());

    context.viewerService().setCurrent2DImagePath(second);
    context.signalBus()->imageListUpdated(
        QStringList{QStringLiteral("a.png"), QStringLiteral("b.png")}, 1);

    QCOMPARE(widget->count(), 2);
    QCOMPARE(widget->currentRow(), 1);
    QVERIFY(!widget->isHidden());
    QCOMPARE(widget->item(0)->toolTip(), first);

    // Selecting a row publishes the full path and asks the viewer to load it.
    widget->setCurrentRow(0);
    QCOMPARE(context.viewerService().imageList(),
             (QStringList{QStringLiteral("a.png"), QStringLiteral("b.png")}));
    QCOMPARE(context.viewerService().imageIndex(), 0);
    QCOMPARE(context.viewerService().getCurrent2DImagePath(), first);
    QCOMPARE(context.viewerService().loadCurrentIndexCalls, 1);

    // Next/prev from the navigator moves the list WITHOUT re-entering the
    // row-changed handler (the signals are blocked): the viewer is the source
    // of truth, and a blocked signal is what keeps the two from looping.
    context.signalBus()->imageIndexChanged(1, 2);
    QCOMPARE(widget->currentRow(), 1);
    QCOMPARE(context.viewerService().loadCurrentIndexCalls, 1);

    // An empty list hides the widget again.
    context.signalBus()->imageListUpdated(QStringList(), -1);
    QVERIFY(widget->isHidden());
}

void TestPluginUi::trainingDockIsBuiltAndRetranslated()
{
    PluginMock::AppContext context(QDir::tempPath());

    // The dock is installed into the shell's central widget by the plugin;
    // here it only has to be built, retranslated and destroyed cleanly.
    auto *dock = new AITrainDockWidget(&context, context.shellService().dialogParent());
    QVERIFY(dock != nullptr);
    QVERIFY(!dock->windowTitle().isEmpty());

    // A language change re-applies the translated texts.
    dock->retranslate();
    QVERIFY(!dock->windowTitle().isEmpty());

    // The process output handler is a no-op while nothing is running — it
    // must not touch the (never started) process.
    QVERIFY(QMetaObject::invokeMethod(dock, "onTrainProcessOutput", Qt::DirectConnection));

    delete dock;
}

// ─── AI inference leaves the GUI thread ─────────────────────────────────────
//
// The plugin's inference entry points used to run the ONNX call inside the GUI
// slot, so the window froze for the whole inference.  The contract pinned here
// is the dispatch: the call happens on another thread, only one runs at a
// time, and a failing backend is reported once instead of killing the process.

void TestPluginUi::detectionInferenceRunsOffTheGuiThread()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString imagePath = writeTestImage(dir);
    QVERIFY(!imagePath.isEmpty());

    context.viewerService().setCurrent2DImagePath(imagePath);
    context.ai().detectionReady = true;

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    QSignalSpy finished(context.signalBus(), &SignalBus::aiInferenceFinished);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));
    QVERIFY2(finished.wait(5000), "the detection result never arrived");

    QCOMPARE(context.ai().detectionCalls, 1);
    QVERIFY2(context.ai().lastInferenceThread != QThread::currentThreadId(),
             "the inference ran on the GUI thread");
    QCOMPARE(context.viewerService().getCurrentAIMode(), AIMode::Detection);
    QCOMPARE(finished.count(), 1);
    QCOMPARE(finished.first().at(0).toString(), QStringLiteral("detection"));
}

void TestPluginUi::segmentationInferenceRunsOffTheGuiThread()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString imagePath = writeTestImage(dir);
    QVERIFY(!imagePath.isEmpty());

    context.viewerService().setCurrent2DImagePath(imagePath);
    context.ai().segmentationReady = true;

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    QSignalSpy finished(context.signalBus(), &SignalBus::aiInferenceFinished);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onSegmentation"));
    QVERIFY2(finished.wait(5000), "the segmentation result never arrived");

    QCOMPARE(context.ai().segmentationCalls, 1);
    QVERIFY2(context.ai().lastInferenceThread != QThread::currentThreadId(),
             "the inference ran on the GUI thread");
    QCOMPARE(context.viewerService().getCurrentAIMode(), AIMode::Segmentation);
    QCOMPARE(finished.first().at(0).toString(), QStringLiteral("segmentation"));
}

void TestPluginUi::oneInferenceAtATime()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString imagePath = writeTestImage(dir);
    QVERIFY(!imagePath.isEmpty());

    context.viewerService().setCurrent2DImagePath(imagePath);
    context.ai().detectionReady = true;
    context.ai().inferenceDelayMs = 1000; // keep the first request in flight

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    QSignalSpy finished(context.signalBus(), &SignalBus::aiInferenceFinished);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));
    // Wait for the worker to enter runDetection before the second click:
    // clicking earlier would race the busy flag and make the assertion below
    // vacuous (the first task might not have started yet either).
    QTRY_COMPARE_WITH_TIMEOUT(context.ai().detectionStarts, 1, 5000);

    // A second click while the first inference is still running: the ONNX
    // session is shared, and two results would race for the same viewer.
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));
    // A dispatched second task enters runDetection almost immediately; give
    // it a bounded window to do so, then assert it never started. The window
    // is shorter than the first inference, so the result is still in flight.
    QTest::qWait(500);
    QCOMPARE(context.ai().detectionStarts, 1);

    QVERIFY2(finished.wait(5000), "the detection result never arrived");
    QCOMPARE(context.ai().detectionCalls, 1);
    QCOMPARE(finished.count(), 1);

    // The busy guard is about "in flight", not a permanent lock: once the
    // result is applied a new request must run again.
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));
    QVERIFY2(finished.wait(5000), "the second detection result never arrived");
    QCOMPARE(context.ai().detectionStarts, 2);
    QCOMPARE(context.ai().detectionCalls, 2);
}

void TestPluginUi::inferenceStaleResultDroppedOnImageChange()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString image1 = writeTestImage(dir, QStringLiteral("frame1.png"));
    const QString image2 = writeTestImage(dir, QStringLiteral("frame2.png"));
    QVERIFY(!image1.isEmpty() && !image2.isEmpty());

    context.viewerService().setCurrent2DImagePath(image1);
    context.ai().detectionReady = true;
    context.ai().inferenceDelayMs = 250;

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    QSignalSpy finished(context.signalBus(), &SignalBus::aiInferenceFinished);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));

    // User or auto-navigation changes image while frame1 is still in flight
    context.viewerService().setCurrent2DImagePath(image2);

    // Frame1 finishes, drops stale result, and starts inference on frame2
    QVERIFY2(finished.wait(5000), "first inference did not finish");
    QVERIFY2(finished.wait(5000), "second inference for frame2 did not finish");

    QCOMPARE(context.ai().detectionCalls, 2);
    QCOMPARE(finished.count(), 2);
}

void TestPluginUi::actionButtonsRemainEnabledDuringNavigation()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString imagePath = writeTestImage(dir);
    QVERIFY(!imagePath.isEmpty());

    context.viewerService().setCurrent2DImagePath(imagePath);
    context.ai().detectionReady = true;
    context.ai().segmentationReady = true;
    context.ai().trackingReady = true;

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    auto *panel = context.shellService().getTabPanel(AppConstants::TabIds::AiProcessor);
    QVERIFY(panel != nullptr);
    const auto buttons = panel->findChildren<QToolButton*>();

    QToolButton *btnDet = nullptr;
    QToolButton *btnSeg = nullptr;
    QToolButton *btnTrack = nullptr;
    for (auto *b : buttons) {
        if (b->text().contains(QStringLiteral("run_detection"))) btnDet = b;
        else if (b->text().contains(QStringLiteral("run_segmentation"))) btnSeg = b;
        else if (b->text().contains(QStringLiteral("video_tracking"))) btnTrack = b;
    }
    QVERIFY(btnDet != nullptr);
    QVERIFY(btnSeg != nullptr);
    QVERIFY(btnTrack != nullptr);

    // Idle state: all buttons enabled
    QVERIFY(btnDet->isEnabled());
    QVERIFY(btnSeg->isEnabled());
    QVERIFY(btnTrack->isEnabled());

    // When navigating or auto-navigating: action buttons must remain enabled to prevent icon flickering
    emit context.signalBus()->autoNavigationChanged(true, true);
    QVERIFY(btnDet->isEnabled());
    QVERIFY(btnSeg->isEnabled());
    QVERIFY(btnTrack->isEnabled());

    // In Detection mode: action buttons also remain enabled (no icon flickering on frame updates)
    context.viewerService().setAIMode(AIMode::Detection);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "updateActions"));
    QVERIFY(btnDet->isEnabled());
    QVERIFY(btnSeg->isEnabled());
    QVERIFY(btnTrack->isEnabled());

    // When auto-navigation stops: buttons remain enabled
    emit context.signalBus()->autoNavigationChanged(false, true);
    QVERIFY(btnDet->isEnabled());
    QVERIFY(btnSeg->isEnabled());
    QVERIFY(btnTrack->isEnabled());
}

void TestPluginUi::anEmptyInferenceResultIsReportedOnce()
{
    // The PRODUCTION failure shape: AIService contains the exception itself
    // (ai::runInferenceGuarded) and returns an EMPTY image — the task does not
    // throw, so blocking::run() has nothing to catch. Before the service
    // returned the INPUT on failure, this path rendered the untouched frame
    // and reported success; the plugin could not tell the two apart.
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString imagePath = writeTestImage(dir);
    QVERIFY(!imagePath.isEmpty());

    context.viewerService().setCurrent2DImagePath(imagePath);
    context.ai().detectionReady = true;
    context.ai().returnEmpty = true;

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    ModalDialogGuard guard;
    guard.start();
    QSignalSpy finished(context.signalBus(), &SignalBus::aiInferenceFinished);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));
    QVERIFY2(finished.wait(5000), "the failure was never reported");
    guard.stop();

    // One completion, the inference really ran, and no stale AI mode: an empty
    // result is a failure, not a frame to render.
    QCOMPARE(finished.count(), 1);
    QCOMPARE(context.ai().detectionCalls, 1);
    QCOMPARE(context.viewerService().getCurrentAIMode(), AIMode::None);
}

void TestPluginUi::aTrackingFailureIsNotReportedAsSuccess()
{
    // VideoTrackerThread emits errorOccurred() and then finishedTracking() —
    // in that order, always. Without the failure flag the second signal shows
    // the "tracking finished" message box, so a session that never tracked a
    // frame ends with a success box on top of the error: the user is told both
    // things at once. Exactly ONE dialog (the error) is the contract.
    PluginMock::AppContext context(QDir::tempPath());
    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    ModalDialogGuard guard;
    guard.start();
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onTrackingError",
                                      Q_ARG(QString, QStringLiteral("the inference failed"))));
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onTrackingFinished"));
    guard.stop();

    QCOMPARE(guard.seenDialogs(), 1);
}

void TestPluginUi::aThrowingInferenceIsReportedOnce()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString imagePath = writeTestImage(dir);
    QVERIFY(!imagePath.isEmpty());

    context.viewerService().setCurrent2DImagePath(imagePath);
    context.ai().detectionReady = true;
    context.ai().throwOnInference = true;

    RunningAIProcessorPlugin instance(context);
    QVERIFY(instance.initialized);

    ModalDialogGuard guard;
    guard.start();
    QSignalSpy finished(context.signalBus(), &SignalBus::aiInferenceFinished);
    QVERIFY(QMetaObject::invokeMethod(&instance.plugin, "onObjectDetection"));
    QVERIFY2(finished.wait(5000), "the failure was never reported");
    guard.stop();

    // blocking::run() contained the exception and delivered a default result:
    // one completion, no stale AI mode, and the process is still alive.
    QCOMPARE(finished.count(), 1);
    QCOMPARE(context.ai().detectionCalls, 0);
    QCOMPARE(context.viewerService().getCurrentAIMode(), AIMode::None);
}


//
// The renderer turns the typed messages into HTML. The contract worth
// pinning: a user message and an assistant message both reach the chat, the
// agent trace shows its steps, and the "thinking" marker sits where the
// caller asked for it.

// A message with the given role and text.
static ChatMessage makeMessage(ChatRole role, const QString &content,
                                const QString &timestamp = QString())
{
    ChatMessage message;
    message.role = role;
    message.content = content;
    message.timestamp = timestamp;
    return message;
}

void TestPluginUi::chatHistoryRendersTheMessages()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTextBrowser browser;

    QList<ChatMessage> history;
    history.append(makeMessage(ChatRole::User, QStringLiteral("xin chào"),
                               QStringLiteral("10:00")));
    history.append(makeMessage(ChatRole::Assistant, QStringLiteral("chào bạn"),
                               QStringLiteral("10:01")));

    ChatMessageRenderer::renderChatHistory(&browser, &context, history, false, -1);

    const QString html = browser.toHtml();
    QVERIFY2(html.contains(QStringLiteral("xin chào")), "the user message is missing");
    QVERIFY2(html.contains(QStringLiteral("chào bạn")), "the assistant message is missing");
    QVERIFY2(html.contains(QStringLiteral("10:00")), "the timestamp is missing");
    // The start-of-chat marker opens every history.
    QVERIFY(html.contains(QStringLiteral("ai.new_chat_start")));
}

void TestPluginUi::theAgentTraceRendersItsSteps()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTextBrowser browser;

    // The agent trace is a message whose content is the server's envelope.
    QJsonObject thinking;
    thinking["type"] = QStringLiteral("thinking");
    thinking["content"] = QStringLiteral("đang suy nghĩ");
    QJsonObject finalAnswer;
    finalAnswer["type"] = QStringLiteral("final_answer");
    finalAnswer["content"] = QStringLiteral("kết quả cuối");
    QJsonArray steps;
    steps.append(thinking);
    steps.append(finalAnswer);
    QJsonObject envelope;
    envelope["steps"] = steps;
    envelope["prior_step_count"] = 0;

    QList<ChatMessage> history;
    history.append(makeMessage(ChatRole::AgentTrace,
                               QString::fromUtf8(QJsonDocument(envelope).toJson(
                                   QJsonDocument::Compact))));

    ChatMessageRenderer::renderChatHistory(&browser, &context, history, false, -1);

    const QString html = browser.toHtml();
    QVERIFY2(html.contains(QStringLiteral("kết quả cuối")),
             "the agent's final answer is missing");
    QVERIFY2(html.contains(QStringLiteral("đang suy nghĩ")),
             "the agent's thinking step is missing");
}

/// One agent step of the given type with the given fields.
static QJsonObject stepOf(const QString &type, const QJsonObject &fields = {})
{
    QJsonObject step = fields;
    step["type"] = type;
    return step;
}

void TestPluginUi::theAgentTraceRendersEveryStepKind()
{
    PluginMock::AppContext context(QDir::tempPath());

    // An earlier response that already RESOLVED an approval request: the
    // renderer scans the whole history for tool_result.action_id, so the
    // approval block for that id must render as "processed", not as a button
    // pair the user can press a second time.
    QJsonArray earlierSteps;
    earlierSteps.append(stepOf("tool_result", {
        {"tool", "read_file"},
        {"action_id", "act-resolved"},
        {"result", QJsonObject{{"path", "src/main.cpp"},
                               {"showing", "1-40"},
                               {"content", "int main() {}"}}}}));
    QJsonObject earlierEnvelope;
    earlierEnvelope["steps"] = earlierSteps;
    earlierEnvelope["prior_step_count"] = 0;

    QJsonArray steps;
    steps.append(stepOf("thinking", {{"content", "<b>suy nghĩ</b>"}}));
    // Two plans: the first draft was rejected and must be folded away.
    steps.append(stepOf("plan", {{"steps", QJsonArray{QStringLiteral("draft one")}}}));
    steps.append(stepOf("plan", {{"steps", QJsonArray{QStringLiteral("bước 1"),
                                                      QStringLiteral("bước 2")}}}));
    // Two reflections: only the final verification outcome is shown.
    steps.append(stepOf("plan_reflection", {
        {"result", QJsonObject{{"passed", false}, {"reason", "draft rejected"}}}}));
    steps.append(stepOf("plan_reflection", {
        {"result", QJsonObject{{"passed", true}, {"reason", "plan verified"}}}}));
    // rag_search is internal noise in the Qt chat: its delegation, call and
    // result are all skipped.
    steps.append(stepOf("delegation", {{"agent", "Researcher"}, {"tool", "rag_search"}}));
    steps.append(stepOf("delegation", {{"agent", "Coder"}, {"tool", "write_file"}}));
    steps.append(stepOf("tool_call", {{"tool", "rag_search"}}));
    steps.append(stepOf("tool_result", {{"tool", "rag_search"}}));
    steps.append(stepOf("verification", {{"tool", "rag_search"},
                                         {"result", QJsonObject{{"passed", true}}}}));
    // A normal tool call, and a tool result with no named renderer (the
    // compact JSON is the fallback).
    steps.append(stepOf("tool_call", {{"tool", "write_file"},
                                      {"params", QJsonObject{{"path", "a.txt"}}}}));
    steps.append(stepOf("tool_result", {{"tool", "unknown_tool"},
                                        {"result", QJsonObject{{"value", 1}}}}));
    // The named tool-result renderers: project status (with more than ten
    // changed files so the "+N more" line appears), directory listing, text
    // search, a citation source, a plain preview, success with and without a
    // path, and an error.
    QJsonArray changed;
    for (int i = 0; i < 12; ++i) changed.append(QStringLiteral("file%1.cpp").arg(i));
    QJsonObject projectStatus;
    projectStatus["git_branch"] = QStringLiteral("main");
    projectStatus["changed_files"] = changed;
    projectStatus["source_file_counts"] = QJsonObject{{"cpp", 3}, {"h", 2}};
    steps.append(stepOf("tool_result", {{"tool", "get_project_status"},
                                        {"result", projectStatus}}));
    steps.append(stepOf("tool_result", {
        {"tool", "list_directory"},
        {"result", QJsonObject{{"path", "src"}, {"count", 12}}}}));
    steps.append(stepOf("tool_result", {
        {"tool", "search_text"},
        {"result", QJsonObject{{"count", 4}, {"query", "TODO"}}}}));
    steps.append(stepOf("tool_result", {
        {"tool", "read_file"},
        {"result", QJsonObject{{"path", "a.cpp"},
                               {"showing", "1-10"},
                               {"presentation", "citation_source"}}}}));
    steps.append(stepOf("tool_result", {
        {"tool", "read_file"},
        {"result", QJsonObject{{"path", "b.cpp"},
                               {"showing", "1-10"},
                               {"content", "int f();"}}}}));
    steps.append(stepOf("tool_result", {
        {"tool", "write_file"},
        {"result", QJsonObject{{"success", true}, {"path", "out.txt"}}}}));
    steps.append(stepOf("tool_result", {
        {"tool", "create_directory"},
        {"result", QJsonObject{{"success", true}}}}));
    steps.append(stepOf("tool_result", {
        {"tool", "delete_file"},
        {"result", QJsonObject{{"error", "permission denied"}}}}));
    // Verification passed and failed.
    steps.append(stepOf("verification", {
        {"tool", "run_command"},
        {"result", QJsonObject{{"passed", true}, {"reason", "exit 0"}}}}));
    steps.append(stepOf("verification", {
        {"tool", "run_command"},
        {"result", QJsonObject{{"passed", false}, {"reason", "exit 1"}}}}));
    // Approvals: one resolved by the earlier tool result, and one pending per
    // description rule (write/patch/create/command) plus the generic tool.
    steps.append(stepOf("pending_approval", {
        {"tool", "write_file"}, {"action_id", "act-resolved"},
        {"params", QJsonObject{{"path", "a.txt"}}}}));
    steps.append(stepOf("pending_approval", {
        {"tool", "write_file"}, {"action_id", "act-pending"},
        {"params", QJsonObject{{"path", "b.txt"}}}}));
    steps.append(stepOf("pending_approval", {
        {"tool", "patch_file"}, {"action_id", "act-patch"},
        {"params", QJsonObject{{"path", "c.txt"}}}}));
    steps.append(stepOf("pending_approval", {
        {"tool", "create_directory"}, {"action_id", "act-dir"},
        {"params", QJsonObject{{"path", "newdir"}}}}));
    steps.append(stepOf("pending_approval", {
        {"tool", "run_command"}, {"action_id", "act-cmd"},
        {"params", QJsonObject{{"command", "dir"}}}}));
    steps.append(stepOf("pending_approval", {
        {"tool", "other_tool"}, {"action_id", "act-other"},
        {"description", "something else"}}));
    steps.append(stepOf("final_answer", {{"content", "xong việc"}}));
    steps.append(stepOf("error", {{"content", "bước lỗi"}}));

    QJsonObject envelope;
    envelope["steps"] = steps;
    envelope["prior_step_count"] = 0;

    // A chatbot handoff is conversational output, not an agent transcript:
    // only the final answer may reach the chat, the tool call must not.
    QJsonArray handoffSteps;
    handoffSteps.append(stepOf("tool_call", {{"tool", "transfer_to_chatbot_agent"}}));
    handoffSteps.append(stepOf("final_answer", {{"content", "câu trả lời hội thoại"}}));
    handoffSteps.append(stepOf("thinking", {{"content", "không được hiện"}}));
    QJsonObject handoffEnvelope;
    handoffEnvelope["steps"] = handoffSteps;
    handoffEnvelope["prior_step_count"] = 0;

    QList<ChatMessage> history;
    history.append(makeMessage(ChatRole::AgentTrace,
                               QString::fromUtf8(QJsonDocument(earlierEnvelope).toJson(
                                   QJsonDocument::Compact))));
    history.append(makeMessage(ChatRole::AgentTrace,
                               QString::fromUtf8(QJsonDocument(envelope).toJson(
                                   QJsonDocument::Compact))));
    history.append(makeMessage(ChatRole::AgentTrace,
                               QString::fromUtf8(QJsonDocument(handoffEnvelope).toJson(
                                   QJsonDocument::Compact))));

    QTextBrowser browser;
    ChatMessageRenderer::renderChatHistory(&browser, &context, history, false, -1);
    const QString html = browser.toHtml();

    // The escaped thinking content, the last plan and the last reflection.
    QVERIFY2(html.contains(QStringLiteral("suy nghĩ")), "thinking missing");
    QVERIFY2(!html.contains(QStringLiteral("draft one")),
             "a superseded plan draft was rendered");
    QVERIFY2(html.contains(QStringLiteral("bước 2")), "the final plan is missing");
    QVERIFY2(!html.contains(QStringLiteral("draft rejected")),
             "a superseded plan reflection was rendered");
    QVERIFY2(html.contains(QStringLiteral("plan verified")), "the final reflection is missing");
    // Delegation, tool call and the named tool results.
    QVERIFY2(html.contains(QStringLiteral("Coder")), "the delegation is missing");
    QVERIFY2(!html.contains(QStringLiteral("Researcher")),
             "a rag_search delegation reached the chat");
    QVERIFY2(html.contains(QStringLiteral("write_file")), "the tool call is missing");
    QVERIFY2(html.contains(QStringLiteral("file9.cpp")), "the changed-file preview is missing");
    // The preview is capped at ten entries; the rest are folded into the
    // "+N more" line (the mock translates the key to itself).
    QVERIFY2(!html.contains(QStringLiteral("file10.cpp")),
             "the changed-file preview is not capped");
    QVERIFY2(html.contains(QStringLiteral("ai.agent_more_items")),
             "the '+N more' line is missing");
    QVERIFY2(html.contains(QStringLiteral("TODO")), "the search query is missing");
    QVERIFY2(html.contains(QStringLiteral("Full source is rendered in the styled citation below.")),
             "the citation-source note is missing");
    QVERIFY2(html.contains(QStringLiteral("int f();")), "the file preview is missing");
    QVERIFY2(html.contains(QStringLiteral("out.txt")), "the written path is missing");
    QVERIFY2(html.contains(QStringLiteral("permission denied")), "the tool error is missing");
    QVERIFY2(html.contains(QStringLiteral("Verification passed")), "verification missing");
    QVERIFY2(html.contains(QStringLiteral("Verification failed")), "failed verification missing");
    // Approvals: the resolved one is marked processed, the pending ones carry
    // their action id and the localized description rule.
    QVERIFY2(html.contains(QStringLiteral("ai.agent_action_processed")),
             "the resolved approval is not marked processed");
    QVERIFY2(html.contains(QStringLiteral("act-pending")), "the pending approval is missing");
    QVERIFY2(html.contains(QStringLiteral("ai.agent_patch_file")), "the patch rule is missing");
    QVERIFY2(html.contains(QStringLiteral("ai.agent_create_directory")),
             "the directory rule is missing");
    QVERIFY2(html.contains(QStringLiteral("ai.agent_run_command")), "the command rule is missing");
    QVERIFY2(html.contains(QStringLiteral("something else")), "the generic approval is missing");
    // Final answer and error step.
    QVERIFY2(html.contains(QStringLiteral("xong việc")), "the final answer is missing");
    QVERIFY2(html.contains(QStringLiteral("bước lỗi")), "the error step is missing");
    // The handoff renders the conversational answer only.
    QVERIFY2(html.contains(QStringLiteral("câu trả lời hội thoại")),
             "the chatbot handoff answer is missing");
    QVERIFY2(!html.contains(QStringLiteral("không được hiện")),
             "the agent trace leaked into a chatbot handoff");
}

void TestPluginUi::userAttachmentsAreRenderedSafely()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    // A readable image: the renderer embeds it as a data: URL because the
    // stored attachment is sealed at rest and the HTML renderer cannot read
    // it from the file system any more.
    const QString png = dir.filePath(QStringLiteral("photo.png"));
    QFile pngFile(png);
    QVERIFY(pngFile.open(QIODevice::WriteOnly));
    pngFile.write(QByteArray("\x89PNG\r\n\x1a\n", 8));
    pngFile.close();

    const QString jpg = dir.filePath(QStringLiteral("photo.jpg"));
    QFile jpgFile(jpg);
    QVERIFY(jpgFile.open(QIODevice::WriteOnly));
    jpgFile.write(QByteArray("\xff\xd8\xff\xe0", 4));
    jpgFile.close();

    // A sealed attachment is ciphertext on disk: a file:// link would hand the
    // OS handler garbage, so only the name may be shown.
    const QString sealed = dir.filePath(QStringLiteral("secret.pdf"));
    QFile sealedFile(sealed);
    QVERIFY(sealedFile.open(QIODevice::WriteOnly));
    sealedFile.write(AtRestCipher::seal(
        "confidential", QString::fromLatin1(FileUtilities::AttachmentContext)));
    sealedFile.close();
    QVERIFY(FileUtilities::isSealedAttachment(sealed));

    // A plain non-image attachment keeps its link.
    const QString plain = dir.filePath(QStringLiteral("notes.txt"));
    QFile plainFile(plain);
    QVERIFY(plainFile.open(QIODevice::WriteOnly));
    plainFile.write("hello");
    plainFile.close();

    // An image that cannot be read falls back to its name — never to a broken
    // image tag.
    const QString missing = dir.filePath(QStringLiteral("gone.png"));

    ChatMessage message = makeMessage(ChatRole::User, QStringLiteral("xem tệp"),
                                      QStringLiteral("not a date"));
    message.attachments = QStringList{png, jpg, sealed, plain, missing};
    // A valid timestamp is formatted for the footer, and reusing an image
    // path takes the data-URL cache (the history is re-rendered on every
    // streamed answer, so re-reading every attachment would be wasteful).
    ChatMessage second = makeMessage(ChatRole::User, QStringLiteral("lại"),
                                     QStringLiteral("2026-10-25 10:00:00"));
    second.attachments = QStringList{png};

    const QList<ChatMessage> history{message, second};
    QTextBrowser browser;
    ChatMessageRenderer::renderChatHistory(&browser, &context, history, false, -1);
    ChatMessageRenderer::renderChatHistory(&browser, &context, history, false, -1);
    const QString html = browser.toHtml();

    QVERIFY2(html.contains(QStringLiteral("data:image/png;base64,")),
             "the PNG attachment is not embedded");
    QVERIFY2(html.contains(QStringLiteral("data:image/jpeg;base64,")),
             "the JPEG attachment is not embedded");
    QVERIFY2(html.contains(QStringLiteral("secret.pdf")), "the sealed name is missing");
    QVERIFY2(html.contains(QStringLiteral("notes.txt")), "the plain attachment is missing");
    QVERIFY2(html.contains(QStringLiteral("gone.png")), "the unreadable name is missing");
    // An unparsable timestamp is shown verbatim instead of an empty footer.
    QVERIFY2(html.contains(QStringLiteral("not a date")), "the raw timestamp is missing");
    // A parsable one is formatted ("MMM d") and the raw value is not shown.
    const QString formatted =
        QDateTime::fromString(QStringLiteral("2026-10-25 10:00:00"),
                              QStringLiteral("yyyy-MM-dd HH:mm:ss")).toString(QStringLiteral("MMM d"));
    QVERIFY2(html.contains(formatted), "the formatted timestamp is missing");
    QVERIFY2(!html.contains(QStringLiteral("2026-10-25 10:00:00")),
             "a parsable timestamp was shown verbatim");
}

void TestPluginUi::theThinkingIndicatorFollowsTheInsertIndex()
{
    PluginMock::AppContext context(QDir::tempPath());
    QTextBrowser browser;

    QList<ChatMessage> history;
    history.append(makeMessage(ChatRole::User, QStringLiteral("câu hỏi")));

    // The marker is inserted after the message at the given index …
    ChatMessageRenderer::renderChatHistory(&browser, &context, history, true, 0);
    QVERIFY(browser.toHtml().contains(QStringLiteral("ai.thinking")));

    // … and at the end of the history when the index is out of range.
    QTextBrowser tail;
    ChatMessageRenderer::renderChatHistory(&tail, &context, history, true, 99);
    QVERIFY(tail.toHtml().contains(QStringLiteral("ai.thinking")));

    // No marker without a pending request.
    QTextBrowser idle;
    ChatMessageRenderer::renderChatHistory(&idle, &context, history, false, -1);
    QVERIFY(!idle.toHtml().contains(QStringLiteral("ai.thinking")));
}

QTEST_MAIN(TestPluginUi)
#include "TestPluginUi.moc"
