#include "ShellService.h"
#include "SignalBus.h"
#include "ILanguageService.h"
#include "CustomProgressDialog.h"

#include <QMap>
#include <QPointer>
#include <QSignalSpy>
#include <QMainWindow>
#include <QDockWidget>
#include <QtTest>

// Fake tối thiểu của ILanguageService — chỉ cần translate() cho fallback
// tra tab theo nhãn hiển thị.
class FakeLanguageService : public ILanguageService {
    Q_OBJECT
public:
    QString currentLanguage() const override { return QStringLiteral("vi"); }
    void setLanguage(const QString&) override {}
    QString translate(const QString& key) const override
    {
        return m_table.value(key, key);
    }
    QString translate(const char* key) const override
    {
        return translate(QString::fromUtf8(key));
    }
    QString displayName() const override { return QStringLiteral("Tiếng Việt"); }
    QString flagEmoji() const override { return QStringLiteral("VN"); }

    void setTranslation(const QString& key, const QString& value)
    {
        m_table.insert(key, value);
    }

private:
    QMap<QString, QString> m_table;
};

class TestShellService : public QObject {
    Q_OBJECT

private slots:
    void defaultsAreNull() {
        SignalBus bus;
        ShellService shell(&bus);
        QVERIFY(shell.menuBar() == nullptr);
        QVERIFY(shell.getProgressDialog() == nullptr);
        QVERIFY(shell.getTabPanel(QStringLiteral("tab.missing")) == nullptr);
        QVERIFY(shell.dialogParent() == nullptr);
    }

    void menuCacheReusesMenusAndRecoversAfterDeletion() {
        SignalBus bus;
        ShellService shell(&bus);
        QWidget parent;
        shell.setMenuParent(&parent);
        QCOMPARE(shell.dialogParent(), &parent);
        QMenuBar bar;
        shell.setMenuBar(&bar);
        QCOMPARE(shell.menuBar(), &bar);

        QMenu* first = shell.getMenu(QStringLiteral("menu.file"));
        QVERIFY(first != nullptr);
        QCOMPARE(first->parentWidget(), &parent);
        QCOMPARE(shell.getMenu(QStringLiteral("menu.file")), first);
        QVERIFY(shell.getMenu(QStringLiteral("menu.edit")) != first);

        // Menu bị hủy ngoài service → QPointer reset, lần get sau tạo lại
        // thay vì trả dangling pointer. Không so sánh địa chỉ với `first`:
        // allocator hoàn toàn có thể cấp lại đúng vùng nhớ vừa giải phóng.
        delete first;
        QMenu* recreated = shell.getMenu(QStringLiteral("menu.file"));
        QVERIFY(recreated != nullptr);
        QCOMPARE(shell.getMenu(QStringLiteral("menu.file")), recreated);
    }

    void tabPanelLookupIsKeyOnly() {
        SignalBus bus;
        ShellService shell(&bus);

        QWidget panel;
        shell.registerTabPanel(QStringLiteral("tab.help"), &panel);
        QCOMPARE(shell.getTabPanel(QStringLiteral("tab.help")), &panel);
        // A translated label is not a routing key: renaming the language must
        // never change which panel a plugin receives.
        QVERIFY(shell.getTabPanel(QStringLiteral("Trợ giúp")) == nullptr);
        QVERIFY(shell.getTabPanel(QStringLiteral("tab.unknown")) == nullptr);
    }

    void deletedPanelIsNotReturned() {
        SignalBus bus;
        ShellService shell(&bus);
        auto* panel = new QWidget;
        shell.registerTabPanel(QStringLiteral("tab.view"), panel);
        QCOMPARE(shell.getTabPanel(QStringLiteral("tab.view")), panel);

        delete panel;
        QVERIFY(shell.getTabPanel(QStringLiteral("tab.view")) == nullptr);
    }

    void updateMenuStatesEmitsStateChanged() {
        SignalBus bus;
        ShellService shell(&bus);
        QSignalSpy spy(&bus, &SignalBus::stateChanged);
        shell.updateMenuStates();
        QCOMPARE(spy.count(), 1);
    }

    void nullBusIsTolerated() {
        // The shell is constructed by AppShell with a bus, but the service
        // must not dereference a missing one during teardown/unit use.
        ShellService shell(nullptr);
        shell.updateMenuStates();
    }

    void nonMainWindowParentIsHandled() {
        SignalBus bus;
        ShellService shell(&bus);
        QWidget parent;
        shell.setMenuParent(&parent);

        QDockWidget dock(QStringLiteral("PlainDock"), &parent);
        shell.addDockWidget(Qt::LeftDockWidgetArea, &dock); // not a QMainWindow: no-op
        QCOMPARE(dock.isVisible(), false);
        QVERIFY(shell.centralWidget() == nullptr);
        shell.deleteDockWidget(nullptr); // documented no-op
    }

    void progressDialogRoundTrip() {
        SignalBus bus;
        FakeLanguageService language;
        ShellService shell(&bus);
        QVERIFY(shell.getProgressDialog() == nullptr);

        CustomProgressDialog dialog(&language);
        shell.setProgressDialog(&dialog);
        QCOMPARE(shell.getProgressDialog(), static_cast<IProgressDialog*>(&dialog));

        // Dialog bị hủy trước service → lookup trả nullptr.
        {
            CustomProgressDialog temporary(&language);
            shell.setProgressDialog(&temporary);
            QCOMPARE(shell.getProgressDialog(), static_cast<IProgressDialog*>(&temporary));
        }
        QVERIFY(shell.getProgressDialog() == nullptr);
    }

    void progressDialogStopSignalIsExposedThroughInterface() {
        FakeLanguageService language;
        CustomProgressDialog dialog(&language);
        IProgressDialog* progress = &dialog;

        QSignalSpy spy(progress, &IProgressDialog::stopRequested);
        QVERIFY(QMetaObject::invokeMethod(progress, "stopRequested"));
        QCOMPARE(spy.count(), 1);
    }

    void addDockWidgetDelegatesToMainWindow() {
        SignalBus bus;
        ShellService shell(&bus);
        QMainWindow window;
        shell.setMenuParent(&window);
        QDockWidget dock(QStringLiteral("TestDock"), &window);
        shell.addDockWidget(Qt::RightDockWidgetArea, &dock);
        QCOMPARE(window.dockWidgetArea(&dock), Qt::RightDockWidgetArea);
    }

    void deleteDockWidgetRemovesAndDestroysIt() {
        SignalBus bus;
        ShellService shell(&bus);
        QMainWindow window;
        shell.setMenuParent(&window);
        auto* dock = new QDockWidget(QStringLiteral("PluginDock"), &window);
        shell.addDockWidget(Qt::RightDockWidgetArea, dock);
        QPointer<QDockWidget> guard(dock);
        shell.deleteDockWidget(dock);
        QVERIFY(guard.isNull());
    }

    void centralWidgetDelegatesToMainWindow() {
        SignalBus bus;
        ShellService shell(&bus);
        QVERIFY(shell.centralWidget() == nullptr);

        QMainWindow window;
        auto* central = new QWidget(&window);
        window.setCentralWidget(central);
        shell.setMenuParent(&window);
        QCOMPARE(shell.centralWidget(), central);
    }

    void setVisibleTogglesParentVisibility() {
        SignalBus bus;
        ShellService shell(&bus);
        QWidget widget;
        widget.hide();
        shell.setMenuParent(&widget);
        shell.setVisible(true);
        QVERIFY(widget.isVisible());
        shell.setVisible(false);
        QVERIFY(!widget.isVisible());
    }

    void registeringNullPanelRemovesTheKey() {
        SignalBus bus;
        ShellService shell(&bus);
        QWidget panel;
        shell.registerTabPanel(QStringLiteral("tab.tmp"), &panel);
        QCOMPARE(shell.getTabPanel(QStringLiteral("tab.tmp")), &panel);

        // Registering a null panel unregisters the key instead of storing a
        // null/expired entry that a lookup would have to filter later.
        shell.registerTabPanel(QStringLiteral("tab.tmp"), nullptr);
        QVERIFY(shell.getTabPanel(QStringLiteral("tab.tmp")) == nullptr);
    }

    void menusWithoutParentAreDeletedWithTheService() {
        SignalBus bus;
        {
            ShellService shell(&bus);
            // No menu bar set → the cached menu has no parent; the service
            // owns and deletes it on destruction.
            QVERIFY(shell.getMenu(QStringLiteral("menu.standalone")) != nullptr);
        }
    }
};

QTEST_MAIN(TestShellService)
#include "TestShellService.moc"
