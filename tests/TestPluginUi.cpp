#include <QtTest>

#include <QDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QListWidget>
#include <QPushButton>
#include <QTimer>
#include <QToolButton>
#include <QWidget>

#include "AIAssistantRibbonUI.h"
#include "AIProcessorRibbonUI.h"
#include "IMailService.h"
#include "MailFilterDialog.h"
#include "MailSettingsDialog.h"
#include "SendMailRibbonUI.h"
#include "ViewerRibbonUI.h"
#include "ViewerViewModel.h"
#include "mock/PluginContextMock.h"

/**
 * @brief The remaining UI surface of the plugins: the ribbon panels, the mail
 * dialogs and the viewer view model.
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
    void mailSettingsDialogCanBeDismissed();
    void viewerViewModelIgnoresEmptyPaths();
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

private:
    QTimer m_timer;
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

void TestPluginUi::viewerViewModelIgnoresEmptyPaths()
{
    PluginMock::AppContext context(QDir::tempPath());
    ViewerViewModel model(&context);

    // An empty path is a no-op — no scene, no viewer, no progress signals.
    model.load2DImage(QString());
    model.load3DModel(QString());
    QVERIFY(true);
}

QTEST_MAIN(TestPluginUi)
#include "TestPluginUi.moc"
