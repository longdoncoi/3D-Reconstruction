#include <QtTest>

#include <QDir>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QPushButton>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QWidget>

#include <iterator>
#include <memory>

#include "AppConfig.h"
#include "AppConstants.h"
#include "IAppContext.h"
#include "ILanguageService.h"
#include "IShellService.h"
#include "MainWindow.h"
#include "ServiceRegistry.h"

/**
 * @brief The real application shell: tab bar, tab panels, menu bar, shutdown.
 *
 * `MainWindow` is compiled into the executable and had no test at all. It is
 * the one class that wires the shell's UI contracts together, and the
 * tab-panel registration is a **plugin contract**: a plugin addresses its
 * panel by tab id (`IShellService::getTabPanel`), so a panel that is built but
 * never registered makes the plugin install nothing — silently, with no
 * error anywhere. This test builds the real window (real `AppShell`, real
 * services) in a sandbox and asserts that contract.
 *
 * The deferred `loadPlugins()` is a `QTimer::singleShot(0, …)`: this test
 * never spins the event loop, so no plugin DLL is loaded (that lifecycle is
 * covered by `TestPluginSmoke`, one plugin per child process).
 */
class TestMainWindow : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();

    void tabBarFollowsTheTabIds();
    void everyTabPanelIsRegisteredInTheShell();
    void clickingATabShowsItsPanel();
    void languageChangeRetranslatesTheTabBar();
    void menuBarAndProgressDialogAreExposedByTheShell();
    void closingTheWindowIsCleanWithoutPlugins();

private:
    QWidget *panelFor(const QString &tabId);
    QPushButton *buttonFor(const QString &tabId);

    std::unique_ptr<MainWindow> m_window;

    static constexpr int kTabCount =
        int(std::size(AppConstants::TabIds::BuildOrder));
};

void TestMainWindow::initTestCase()
{
    // Sandbox every path the services may read or write: a sandbox directory
    // without a CMakeLists.txt parent puts AppConfig in "production" mode, so
    // the data root resolves under the test-mode AppData location instead of
    // the repository tree (same setup as TestAppShell).
    QStandardPaths::setTestModeEnabled(true);
    const QString sandbox =
        QDir::tempPath() + QStringLiteral("/3d_recon_mainwindow");
    QDir().mkpath(sandbox);
    QT_WARNING_PUSH
    QT_WARNING_DISABLE_DEPRECATED
    AppConfig::bootstrap().initialize(sandbox);
    QT_WARNING_POP

    m_window = std::make_unique<MainWindow>();
    QVERIFY(m_window->context() != nullptr);
}

void TestMainWindow::cleanupTestCase()
{
    m_window.reset();
}

QWidget *TestMainWindow::panelFor(const QString &tabId)
{
    return m_window->findChild<QWidget *>(QStringLiteral("tabPanel_") + tabId);
}

QPushButton *TestMainWindow::buttonFor(const QString &tabId)
{
    const auto buttons =
        m_window->findChildren<QPushButton *>(QStringLiteral("tabHeaderBtn"));
    for (QPushButton *button : buttons) {
        if (button->property("tabKey").toString() == tabId)
            return button;
    }
    return nullptr;
}

void TestMainWindow::tabBarFollowsTheTabIds()
{
    const auto buttons =
        m_window->findChildren<QPushButton *>(QStringLiteral("tabHeaderBtn"));
    QCOMPARE(buttons.size(), kTabCount);

    // The order in the title bar is the single source of truth (BuildOrder),
    // not an accident of widget creation order.
    auto *bar = m_window->findChild<QWidget *>(QStringLiteral("tabHeaderBar"));
    QVERIFY(bar != nullptr);
    auto *layout = qobject_cast<QHBoxLayout *>(bar->layout());
    QVERIFY(layout != nullptr);
    QCOMPARE(layout->count(), kTabCount);

    for (int i = 0; i < kTabCount; ++i) {
        const QString id =
            QString::fromLatin1(AppConstants::TabIds::BuildOrder[i]);
        auto *button = qobject_cast<QPushButton *>(layout->itemAt(i)->widget());
        QVERIFY(button != nullptr);
        QCOMPARE(button->property("tabKey").toString(), id);
        // The id is also the translation key: the label is translated, the
        // routing key is not (renaming a language can never move a panel).
        QCOMPARE(button->text(), m_window->context()->translate(id));
    }
}

void TestMainWindow::everyTabPanelIsRegisteredInTheShell()
{
    IShellService *shell = m_window->context()->shell();
    QVERIFY(shell != nullptr);

    auto *stack =
        m_window->findChild<QStackedWidget *>(QStringLiteral("tabContentStrip"));
    QVERIFY(stack != nullptr);
    QCOMPARE(stack->count(), kTabCount);

    for (const char *tabId : AppConstants::TabIds::BuildOrder) {
        const QString id = QString::fromLatin1(tabId);
        QWidget *panel = panelFor(id);
        QVERIFY2(panel != nullptr,
                 qPrintable(QStringLiteral("no panel built for %1").arg(id)));
        // THE PLUGIN CONTRACT: a plugin looks the panel up by tab id. A panel
        // that exists in the widget tree but is not registered is invisible
        // to every plugin, and nothing reports it.
        QCOMPARE(shell->getTabPanel(id), panel);
        // The panel is also the one the stack owns (not a look-alike).
        QVERIFY(stack->indexOf(panel) >= 0);
    }
}

void TestMainWindow::clickingATabShowsItsPanel()
{
    auto *stack =
        m_window->findChild<QStackedWidget *>(QStringLiteral("tabContentStrip"));
    QVERIFY(stack != nullptr);

    const QString first =
        QString::fromLatin1(AppConstants::TabIds::BuildOrder[0]);
    const QString last =
        QString::fromLatin1(AppConstants::TabIds::BuildOrder[kTabCount - 1]);

    // The window opens on the first tab.
    QCOMPARE(stack->currentWidget(), panelFor(first));

    QPushButton *lastButton = buttonFor(last);
    QVERIFY(lastButton != nullptr);
    lastButton->click();

    QCOMPARE(stack->currentWidget(), panelFor(last));
    QVERIFY(lastButton->isChecked());
    QVERIFY(!buttonFor(first)->isChecked());
}

void TestMainWindow::languageChangeRetranslatesTheTabBar()
{
    auto *language =
        m_window->context()->services()->getRequired<ILanguageService>();
    QVERIFY(language != nullptr);

    const QString id = QString::fromLatin1(AppConstants::TabIds::BuildOrder[0]);
    const QString before = buttonFor(id)->text();

    const QString target = language->currentLanguage() == QStringLiteral("vi")
                               ? QStringLiteral("en")
                               : QStringLiteral("vi");
    language->setLanguage(target);

    for (const char *tabId : AppConstants::TabIds::BuildOrder) {
        const QString key = QString::fromLatin1(tabId);
        QCOMPARE(buttonFor(key)->text(), language->translate(key));
    }

    // Non-vacuity: the two languages differ for this key, so a missing
    // connection (the label keeping the old language) fails here.
    QVERIFY2(buttonFor(id)->text() != before,
             "the tab label did not follow the language change");
}

void TestMainWindow::menuBarAndProgressDialogAreExposedByTheShell()
{
    IShellService *shell = m_window->context()->shell();
    QVERIFY(shell != nullptr);

    // Plugins add their menus to THIS bar (not QMainWindow::menuBar(), which
    // is unused by the frameless title bar).
    auto *menuBar =
        m_window->findChild<QMenuBar *>(QStringLiteral("integratedMenuBar"));
    QVERIFY(menuBar != nullptr);
    QCOMPARE(shell->menuBar(), menuBar);
    QCOMPARE(shell->dialogParent(), m_window.get());

    // The progress dialog is owned by the shell, so a plugin can never
    // outlive it (the old contract leaked the concrete widget into the ABI).
    QVERIFY(shell->getProgressDialog() != nullptr);
}

void TestMainWindow::closingTheWindowIsCleanWithoutPlugins()
{
    // The deferred loadPlugins() never ran (no event loop), so the plugin
    // host is empty and shutdownPlugins() takes the empty path. Closing must
    // still be clean; the destructor then tears the service graph down while
    // the widget tree is still alive.
    QVERIFY(m_window->close());
    QVERIFY(!m_window->isVisible());
}

QTEST_MAIN(TestMainWindow)

#include "TestMainWindow.moc"
