#include <QtTest>

#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QTableWidget>
#include <QPushButton>
#include <QStackedWidget>
#include <QTemporaryDir>
#include <QToolButton>
#include <QWidget>

#include "AboutDialog.h"
#include "AdminUserManagerDialog.h"
#include "AvatarCropperDialog.h"
#include "ChangePasswordDialog.h"
#include "ForgotPasswordDialog.h"
#include "LicenseActivationDialog.h"
#include "LoginDialog.h"
#include "SettingsDialog.h"
#include "ThemeSelectionDialog.h"
#include "OverlayResizer.h"
#include "UserAuthPlugin.h"
#include "mock/PluginContextMock.h"

/**
 * @brief The authentication dialogs of the user-auth plugin.
 *
 * These dialogs are the biggest GUI blocks of the plugin layer (≈1 000
 * lines). Each one receives the user service through its CONSTRUCTOR, so the
 * test builds them directly — no plugin instance, no process-global binding.
 * (They used to read `UserAuthPlugin::users()`, a static the plugin bound in
 * `initialize()`, which forced this test to initialize the whole plugin just
 * to construct a dialog.)
 *
 * The dialogs are never shown: construction is what builds their widget tree
 * (and that is where the layout/wiring bugs live).
 */
class TestUserAuthDialogs : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void loginDialogIsBuilt();
    void aboutDialogShowsTheUser();
    void changePasswordDialogIsBuilt();
    void forgotPasswordDialogIsBuilt();
    void forgotPasswordDialogGuardsTheReset();
    void backgroundOverlayFollowsTheWindow();
    void themeSelectionDelegatesTheBackgroundToItsOwner();
    void settingsDialogTestsTheMailConnection();
    void licenseActivationDialogIsBuilt();
    void settingsDialogIsBuilt();
    void adminUserManagerDialogIsBuilt();
    void avatarCropperDialogIsBuilt();
    void userAuthPluginLifecycle();

private:
    QTemporaryDir m_root;
    PluginMock::AppContext *m_context = nullptr;
};

void TestUserAuthDialogs::initTestCase()
{
    QVERIFY(m_root.isValid());

    m_context = new PluginMock::AppContext(m_root.path());
    m_context->users().signIn(QStringLiteral("tester"));
}

void TestUserAuthDialogs::loginDialogIsBuilt()
{
    LoginDialog dialog(&m_context->users());
    // ModernDialog draws its own title bar: the title lives in a label.
    auto *title = dialog.findChild<QLabel *>(QStringLiteral("modernTitleLabel"));
    QVERIFY(title != nullptr);
    QVERIFY(!title->text().isEmpty());
    // The dialog must offer the essential fields.
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::aboutDialogShowsTheUser()
{
    AboutDialog dialog(QStringLiteral("tester"), &m_context->users());
    auto *title = dialog.findChild<QLabel *>(QStringLiteral("modernTitleLabel"));
    QVERIFY(title != nullptr);
    QVERIFY(!dialog.findChildren<QLabel *>().isEmpty());
}

void TestUserAuthDialogs::changePasswordDialogIsBuilt()
{
    ChangePasswordDialog dialog(QStringLiteral("tester"), &m_context->users());
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::forgotPasswordDialogIsBuilt()
{
    ForgotPasswordDialog dialog(&m_context->users());
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::settingsDialogTestsTheMailConnection()
{
    SettingsDialog dialog(QStringLiteral("tester"), false, m_context, &m_context->users());

    // The mail page is the last one in the navigation list.
    auto *nav = dialog.findChild<QListWidget *>(QStringLiteral("settingsNavList"));
    QVERIFY(nav != nullptr);
    QVERIFY(nav->count() > 0);
    nav->setCurrentRow(nav->count() - 1);

    // The "test connection" button runs the SMTP probe off the GUI thread.
    auto *testButton =
        dialog.findChild<QPushButton *>(QStringLiteral("mailTestConnectionButton"));
    QVERIFY(testButton != nullptr);
    testButton->click();

    // The stub service reports a failure; the dialog must survive it (the
    // background task is waited for in the destructor, with a bound).
    QVERIFY(dialog.isEnabled());
}

void TestUserAuthDialogs::backgroundOverlayFollowsTheWindow()
{
    // The background overlay is installed on the shell window and follows its
    // size. The event filter that does it is a NAMED class, and the OWNER's
    // slot is the only state: the filter is created with the overlay and
    // released with it. There is no second source of truth (a widget
    // property, a flag) that could drift away from the filter that is
    // actually installed — that drift is what used to leave a filter behind
    // with no owner to remove it.
    QWidget shell;
    shell.resize(800, 600);
    OverlayResizer *filter = nullptr;

    // Without a background nothing is installed at all.
    OverlayResizer::applyTo(&shell, QString(), 30, false, filter);
    QVERIFY(filter == nullptr);
    QVERIFY(shell.findChild<QLabel *>(QStringLiteral("bgGlobalOverlay")) == nullptr);

    // With a background the overlay and its resizer are created.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString image = dir.path() + QStringLiteral("/background.png");
    QPixmap pixmap(64, 64);
    pixmap.fill(Qt::darkBlue);
    QVERIFY(pixmap.save(image));

    OverlayResizer::applyTo(&shell, image, 30, true, filter);
    QVERIFY(filter != nullptr);
    auto *overlay = shell.findChild<QLabel *>(QStringLiteral("bgGlobalOverlay"));
    QVERIFY(overlay != nullptr);
    QCOMPARE(overlay->geometry(), shell.rect());

    // A resize of the window is propagated to the overlay by the filter.
    shell.resize(640, 480);
    QEvent resize(QEvent::Resize);
    filter->eventFilter(&shell, &resize);
    QCOMPARE(overlay->geometry(), shell.rect());

    // Applying it a second time does not stack a second filter: the slot
    // still holds the one that is installed.
    OverlayResizer *sameFilter = filter;
    OverlayResizer::applyTo(&shell, image, 30, true, filter);
    QCOMPARE(filter, sameFilter);
    QCOMPARE(shell.findChildren<OverlayResizer *>().size(), 1);

    // Switching the background off releases the filter as well — the shell
    // must not keep calling into this DLL for a window that has no overlay.
    OverlayResizer::applyTo(&shell, image, 30, false, filter);
    QVERIFY(filter == nullptr);
    QCOMPARE(shell.findChildren<OverlayResizer *>().size(), 0);

    // A path that no longer resolves to an image must not leave the previous
    // picture on screen pretending to be current: the overlay is hidden (and
    // the filter, which is still needed to follow resizes, stays installed).
    OverlayResizer::applyTo(&shell, image, 30, true, filter);
    QVERIFY(filter != nullptr);
    QVERIFY(!overlay->isHidden());
    OverlayResizer::applyTo(&shell, dir.path() + QStringLiteral("/gone.png"), 30, true, filter);
    QVERIFY(filter != nullptr);
    QVERIFY(overlay->isHidden());

    // And the owner can release it explicitly before the DLL is unloaded.
    OverlayResizer::applyTo(&shell, image, 30, true, filter);
    QVERIFY(filter != nullptr);
    shell.removeEventFilter(filter);
    delete filter;
    filter = nullptr;
    QVERIFY(shell.findChildren<OverlayResizer *>().isEmpty());
}

void TestUserAuthDialogs::forgotPasswordDialogGuardsTheReset(){
    ForgotPasswordDialog dialog(&m_context->users());
    // The three steps of the dialog: e-mail, OTP, new password + confirmation.
    const QList<QLineEdit *> edits = dialog.findChildren<QLineEdit *>();
    QVERIFY(edits.size() >= 4);
    QLineEdit *newPass = edits.at(2);
    QLineEdit *confirmPass = edits.at(3);
    auto *status = dialog.findChild<QLabel *>(QStringLiteral("statusLabel"));
    QVERIFY(status != nullptr);
    // The dialog itself is never shown, so the explicit hidden flag is the
    // observable (isVisible() would also require the whole chain to be shown).
    QVERIFY(status->isHidden());

    // A mismatching confirmation is refused before anything is written.
    newPass->setText(QStringLiteral("Secret1!"));
    confirmPass->setText(QStringLiteral("Other1!"));
    QVERIFY(QMetaObject::invokeMethod(&dialog, "onResetPassword", Qt::DirectConnection));
    QVERIFY(!status->isHidden());
    QVERIFY(!status->text().isEmpty());

    // A matching confirmation reaches the service — which refuses it here,
    // because the dialog has no verified proof of ownership. The reason is
    // shown instead of a silent no-op.
    status->setVisible(false);
    confirmPass->setText(QStringLiteral("Secret1!"));
    QVERIFY(QMetaObject::invokeMethod(&dialog, "onResetPassword", Qt::DirectConnection));
    QVERIFY(!status->isHidden());
    QVERIFY(!status->text().isEmpty());
}

void TestUserAuthDialogs::licenseActivationDialogIsBuilt()
{
    LicenseActivationDialog dialog(QStringLiteral("tester"), &m_context->users(), true);
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
}

void TestUserAuthDialogs::settingsDialogIsBuilt()
{
    SettingsDialog dialog(QStringLiteral("tester"), true, m_context, &m_context->users());
    // The settings dialog is a navigation stack: the list plus the pages.
    QVERIFY(!dialog.findChildren<QListWidget *>().isEmpty());
    QVERIFY(!dialog.findChildren<QStackedWidget *>().isEmpty());
    QVERIFY(!dialog.findChildren<QToolButton *>().isEmpty());

    auto *navList = dialog.findChild<QListWidget *>(QStringLiteral("settingsNavList"));
    if (navList) {
        for (int i = 0; i < navList->count(); ++i) {
            emit navList->itemClicked(navList->item(i));
        }
    }

    const auto buttons = dialog.findChildren<QPushButton *>();
    for (auto *btn : buttons) {
        if (btn->objectName() == QStringLiteral("primary") ||
            btn->text() == UserAuthPlugin::translate("common.apply")) {
            btn->click();
            break;
        }
    }
}

void TestUserAuthDialogs::adminUserManagerDialogIsBuilt()
{
    AdminUserManagerDialog dialog(&m_context->users());
    // The users and the licence keys are listed in tables.
    QVERIFY(!dialog.findChildren<QTableWidget *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::avatarCropperDialogIsBuilt()
{
    // No avatar file exists in the stub path: the dialog has to cope with it.
    AvatarCropperDialog dialog(QStringLiteral("tester"), &m_context->paths(),
                              &m_context->users(), nullptr);
    QVERIFY(dialog.findChildren<QPushButton *>().size() >= 1);
}

void TestUserAuthDialogs::userAuthPluginLifecycle()
{
    UserAuthPlugin plugin;
    QCOMPARE(plugin.pluginName(), QStringLiteral("User Authentication"));
    QCOMPARE(plugin.loadOrder(), 5);
    m_context->users().signIn(QStringLiteral("tester"));
    QVERIFY(plugin.initialize(m_context));
    emit m_context->signalBus()->agentUiActionRequested(
        QStringLiteral("language.change"),
        QVariantMap{{QStringLiteral("language"), QStringLiteral("en")},
                    {QStringLiteral("request_id"), QStringLiteral("req-1")}});
    emit m_context->signalBus()->languageChanged(QStringLiteral("en"));
    plugin.cleanup();
}

QTEST_MAIN(TestUserAuthDialogs)
#include "TestUserAuthDialogs.moc"

void TestUserAuthDialogs::themeSelectionDelegatesTheBackgroundToItsOwner()
{
    // The dialog is a *form*: it collects the choice and hands it to the
    // plugin, which owns the overlay and the event filter. It must never
    // install a filter itself — the filter outlives this widget and its vtable
    // lives in the plugin DLL.
    ThemeSelectionDialog dialog(QStringLiteral("tester"), &m_context->users());
    QVERIFY(dialog.findChildren<OverlayResizer *>().isEmpty());

    int calls = 0;
    QString appliedPath = QStringLiteral("<not called>");
    int appliedOpacity = -1;
    bool appliedEnabled = true;
    dialog.setBackgroundRequest([&](const QString &path, int opacity, bool enabled) {
        ++calls;
        appliedPath = path;
        appliedOpacity = opacity;
        appliedEnabled = enabled;
    });

    // ── Choosing a colour marks exactly that option ────────────────────────
    const QList<QToolButton *> options =
        dialog.findChildren<QToolButton *>(QStringLiteral("themeOption"));
    QCOMPARE(options.size(), 6);
    QToolButton *emerald = nullptr;
    for (auto *option : options) {
        if (option->text() == QStringLiteral("Emerald")) emerald = option;
    }
    QVERIFY(emerald != nullptr);
    QCOMPARE(emerald->property("selected").toBool(), false);
    emerald->click();
    QCOMPARE(emerald->property("selected").toBool(), true);
    int selected = 0;
    for (auto *option : options) {
        if (option->property("selected").toBool()) ++selected;
    }
    QCOMPARE(selected, 1);

    // ── Apply is delegated, not performed here ─────────────────────────────
    auto *apply = dialog.findChild<QPushButton *>(QStringLiteral("primary"));
    QVERIFY(apply != nullptr);
    apply->click();

    QCOMPARE(calls, 1);
    QCOMPARE(appliedPath, QString());
    QCOMPARE(appliedEnabled, false);
    QVERIFY(appliedOpacity >= 0);
    QVERIFY(appliedOpacity <= 100);

    // The dialog installed nothing on anything: no filter, no overlay.
    QVERIFY(dialog.findChildren<OverlayResizer *>().isEmpty());
    QVERIFY(dialog.findChild<QLabel *>(QStringLiteral("bgGlobalOverlay")) == nullptr);
}
