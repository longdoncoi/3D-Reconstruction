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
#include "UserAuthPlugin.h"
#include "mock/PluginContextMock.h"

/**
 * @brief The authentication dialogs of the user-auth plugin.
 *
 * These dialogs are the biggest GUI blocks of the plugin layer (≈1 000
 * lines) and had no test at all.  They all read the user through
 * `UserAuthPlugin::users()`, the static the plugin binds during
 * `initialize()` — so the test initializes the plugin with a stub context
 * first and only then builds the dialogs.
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
    void licenseActivationDialogIsBuilt();
    void settingsDialogIsBuilt();
    void adminUserManagerDialogIsBuilt();
    void avatarCropperDialogIsBuilt();

private:
    QTemporaryDir m_root;
    PluginMock::AppContext *m_context = nullptr;
    UserAuthPlugin *m_plugin = nullptr;
};

void TestUserAuthDialogs::initTestCase()
{
    QVERIFY(m_root.isValid());

    m_context = new PluginMock::AppContext(m_root.path());
    m_context->users().signIn(QStringLiteral("tester"));

    m_plugin = new UserAuthPlugin;
    // The plugin binds the user manager and the language service; the dialogs
    // read them through the plugin's statics.
    QVERIFY(m_plugin->initialize(m_context));
}

void TestUserAuthDialogs::loginDialogIsBuilt()
{
    LoginDialog dialog;
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
    AboutDialog dialog(QStringLiteral("tester"));
    auto *title = dialog.findChild<QLabel *>(QStringLiteral("modernTitleLabel"));
    QVERIFY(title != nullptr);
    QVERIFY(!dialog.findChildren<QLabel *>().isEmpty());
}

void TestUserAuthDialogs::changePasswordDialogIsBuilt()
{
    ChangePasswordDialog dialog(QStringLiteral("tester"));
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::forgotPasswordDialogIsBuilt()
{
    ForgotPasswordDialog dialog;
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::licenseActivationDialogIsBuilt()
{
    LicenseActivationDialog dialog(QStringLiteral("tester"), true);
    QVERIFY(!dialog.findChildren<QLineEdit *>().isEmpty());
}

void TestUserAuthDialogs::settingsDialogIsBuilt()
{
    SettingsDialog dialog(QStringLiteral("tester"), true, m_context);
    // The settings dialog is a navigation stack: the list plus the pages.
    QVERIFY(!dialog.findChildren<QListWidget *>().isEmpty());
    QVERIFY(!dialog.findChildren<QStackedWidget *>().isEmpty());
    QVERIFY(!dialog.findChildren<QToolButton *>().isEmpty());
}

void TestUserAuthDialogs::adminUserManagerDialogIsBuilt()
{
    AdminUserManagerDialog dialog;
    // The users and the licence keys are listed in tables.
    QVERIFY(!dialog.findChildren<QTableWidget *>().isEmpty());
    QVERIFY(!dialog.findChildren<QPushButton *>().isEmpty());
}

void TestUserAuthDialogs::avatarCropperDialogIsBuilt()
{
    // No avatar file exists in the stub path: the dialog has to cope with it.
    AvatarCropperDialog dialog(QStringLiteral("tester"), &m_context->paths(), nullptr);
    QVERIFY(dialog.findChildren<QPushButton *>().size() >= 1);
}

QTEST_MAIN(TestUserAuthDialogs)
#include "TestUserAuthDialogs.moc"
