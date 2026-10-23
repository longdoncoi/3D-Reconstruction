#include <QtTest>
#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QStringList>
#include <QTemporaryDir>
#include <QTimer>

#include "IPlugin.h"
#include "PluginHost.h"
#include "mock/PluginContextMock.h"

/**
 * @brief Smoke test on the real plugin binaries.
 *
 * Unlike TestPluginHost (which injects a fake loader), this test exercises
 * QPluginLoader against the DLLs produced by the default build: a broken
 * Q_PLUGIN_METADATA block, a missing export, a stale IPlugin ABI or invalid
 * ordering metadata is caught before the application is even started.
 *
 * The plugin directory is injected at compile time by CMake
 * (PLUGIN_SMOKE_DIR, resolved from the ViewerPlugin target) and the test target
 * depends on every plugin target so the DLLs exist when CTest runs.
 *
 * ## Plugin initialization
 *
 * The second test initializes and cleans up every shipped plugin against a
 * stub context (tests/core/mock/PluginContextMock.h).  Each plugin runs in its
 * **own child process**: the plugins are GUI shells that install themselves
 * into the shell's widget tree, and letting them share one process leaks
 * widget ownership from one plugin into the next (the widget of plugin A is
 * still parented to a shell widget when plugin B tears its own tree down).
 * The child process keeps that contamination out of the test binary and
 * turns a crash in one plugin into a reported failure instead of a dead
 * suite.  The child is this very executable, re-run with
 * ``--init-plugin <name>``.
 */
class TestPluginSmoke : public QObject {
    Q_OBJECT

private slots:
    void allShippedPluginsLoadAndResolve();
    void everyPluginInitializesAndCleansUp();
};

namespace {

QString pluginDirectory()
{
#ifdef PLUGIN_SMOKE_DIR
    return QStringLiteral(PLUGIN_SMOKE_DIR);
#else
    return QString();
#endif
}

/// Closes a modal dialog as soon as it appears, so a plugin that asks the
/// user for something (login, licence) cannot hang the child process.
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

/// Writes a diagnostic for a failed child run. On Windows a GUI process has
/// no console, so the parent test cannot always read the warning from the
/// child's stderr — the file keeps the reason next to the failure report.
void writeLeftoverReport(const QString &pluginName, const QString &reason)
{
    QFile report(QDir::temp().filePath(
        QStringLiteral("3dreconstruction-plugin-leftovers-%1.txt")
            .arg(QString::fromLatin1(pluginName.toUtf8().toBase64().toPercentEncoding()))));
    if (!report.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    report.write(QStringLiteral("%1: %2").arg(pluginName, reason).toUtf8());
    report.close();
}

/// Child-process entry point: initialize and clean up one plugin.
int runPluginInitialization(const QString &pluginName)
{
    QTemporaryDir root;
    if (!root.isValid()) {
        return 5;
    }
    // The context is declared BEFORE the host on purpose: the host unloads
    // the plugin DLLs in its destructor, and that has to happen while the
    // shell's widget tree is still alive — the same order as the real
    // application (MainWindow::shutdownPlugins).
    PluginMock::AppContext context(root.path());
    context.users().signIn(QStringLiteral("tester"));
    QStringList hostErrors;
    int releasedPlugins = 0;
    int loadedPlugins = 0;

    {
        PluginHost host;
        host.discover(pluginDirectory());
        if (!host.errors().isEmpty()) {
            qWarning() << "discovery failed:" << host.errors();
            return 2;
        }
        if (!host.resolveOrder()) {
            qWarning() << "resolve failed:" << host.errors();
            return 3;
        }
        loadedPlugins = host.plugins().size();
        IPlugin *target = nullptr;
        for (IPlugin *plugin : host.plugins()) {
            if (plugin->pluginName() == pluginName) {
                target = plugin;
                break;
            }
        }
        if (!target) {
            qWarning() << "plugin not found:" << pluginName;
            return 4;
        }

        ModalDialogGuard modalGuard;
        modalGuard.start();
        const bool initialized = target->initialize(&context);
        modalGuard.stop();
        if (!initialized) {
            // A plugin may legitimately decline (e.g. no licence); that is not
            // a crash and not a failure of the plugin's wiring.
            qInfo() << "plugin declined initialization:" << pluginName;
            return 0;
        }

        // A plugin has to keep its installed UI usable after a language
        // change.  Emit the real application signal before cleanup so every
        // plugin-owned translation callback is exercised while its widgets
        // still exist.
        emit context.signalBus()->languageChanged(QStringLiteral("en"));
        target->cleanup();

        // The widget-ownership contract: a plugin must release every object
        // it installed into the shell. An object that outlives the plugin
        // would dispatch a virtual call into the DLL the host is about to
        // unload.
        const QStringList leftovers = context.shellService().leftoverShellObjects();
        if (!leftovers.isEmpty()) {
            qWarning() << "plugin left" << leftovers << "in the shell";
            writeLeftoverReport(pluginName, QStringLiteral("left %1 object(s) in the shell:\n  - %2")
                                     .arg(leftovers.size())
                                     .arg(leftovers.join(QStringLiteral("\n  - "))));
            return 6;
        }
        // The check must not be vacuous: a plugin that installed nothing into
        // the shell would trivially leave nothing behind.
        const int installed = context.shellService().installedShellObjectCount();
        if (installed == 0) {
            qWarning() << "plugin installed nothing into the shell";
            writeLeftoverReport(pluginName,
                                QStringLiteral("installed nothing into the shell — "
                                               "the ownership check would be vacuous\n"));
            return 7;
        }

        // The explicit shutdown releases the plugins; the destructor repeats
        // it (it is idempotent). The errors of the release are collected
        // here, where the host is still alive.
        host.shutdown();
        hostErrors = host.takeErrors();
        releasedPlugins = host.releasedPluginCount();
    }
    // The host is gone: it must have released the plugin objects — a plugin
    // whose root is still alive keeps its whole QObject tree, and every
    // object it installed in the shell.
    if (!hostErrors.isEmpty() || releasedPlugins != loadedPlugins) {
        qWarning() << "the plugin was not released:" << hostErrors
                   << releasedPlugins << "of" << loadedPlugins;
        writeLeftoverReport(pluginName,
                            QStringLiteral("the plugin was not released: %1 of %2\n%3")
                                .arg(releasedPlugins).arg(loadedPlugins)
                                .arg(hostErrors.join(QStringLiteral("\n"))));
        return 8;
    }
    return 0;
}
} // namespace

void TestPluginSmoke::allShippedPluginsLoadAndResolve()
{
    const QString directory = pluginDirectory();
    QVERIFY2(!directory.isEmpty(), "PLUGIN_SMOKE_DIR was not configured by CMake");
    QVERIFY2(QDir(directory).exists(),
             qPrintable(QStringLiteral("plugin directory not found: ") + directory));

    PluginHost host;
    host.discover(directory);
    QVERIFY2(host.errors().isEmpty(),
             qPrintable(host.errors().join(QStringLiteral(" | "))));

    QStringList names;
    for (IPlugin *plugin : host.plugins()) {
        QCOMPARE(plugin->apiVersion(), IPlugin::kApiVersion);
        names << plugin->pluginName();
    }
    names.sort();

    QStringList expected{"AI Assistant Plugin", "AI Processor Plugin",
                         "Reconstruction Plugin", "Send Mail Plugin",
                         "User Authentication Plugin", "Viewer Plugin"};
    expected.sort();
    QCOMPARE(names, expected);

    // The dependency/phase metadata of the real binaries must resolve cleanly.
    QVERIFY2(host.resolveOrder(),
             qPrintable(host.errors().join(QStringLiteral(" | "))));

    // Host teardown (~PluginHost → shutdown()) must be safe even though the
    // plugins were never initialized.
}

void TestPluginSmoke::everyPluginInitializesAndCleansUp()
{
    const QString directory = pluginDirectory();
    QVERIFY2(!directory.isEmpty(), "PLUGIN_SMOKE_DIR was not configured by CMake");

    PluginHost host;
    host.discover(directory);
    QVERIFY2(host.errors().isEmpty(),
             qPrintable(host.errors().join(QStringLiteral(" | "))));
    QVERIFY2(host.resolveOrder(),
             qPrintable(host.errors().join(QStringLiteral(" | "))));

    const QString self = QCoreApplication::applicationFilePath();
    QStringList failures;
    for (IPlugin *plugin : host.plugins()) {
        const QString name = plugin->pluginName();
        QProcess child;
        // The child's diagnostics are merged into one channel so a failure
        // report can name the objects the plugin left behind.
        child.setProcessChannelMode(QProcess::MergedChannels);
        child.start(self, {QStringLiteral("--init-plugin"), name});
        if (!child.waitForStarted(30000)) {
            failures << QStringLiteral("%1: child did not start").arg(name);
            continue;
        }
        if (!child.waitForFinished(120000)) {
            child.kill();
            child.waitForFinished(5000);
            failures << QStringLiteral("%1: timed out").arg(name);
            continue;
        }
        if (child.exitStatus() != QProcess::NormalExit || child.exitCode() != 0) {
            const QString output = QString::fromUtf8(child.readAll()).trimmed();
            failures << QStringLiteral("%1: exit code %2 (%3)%4")
                             .arg(name)
                             .arg(child.exitCode())
                             .arg(child.exitStatus() == QProcess::CrashExit
                                      ? QStringLiteral("crashed")
                                      : QStringLiteral("abnormal exit"))
                             .arg(output.isEmpty() ? QString()
                                                  : QStringLiteral(" — ") + output);
        }
    }

    // Every shipped plugin must survive a real initialize()/cleanup() cycle:
    // a plugin that crashes here would take the whole application down at
    // start-up.
    QVERIFY2(failures.isEmpty(), qPrintable(failures.join(QStringLiteral("; "))));
}

int main(int argc, char *argv[])
{
    if (argc > 2 && QByteArray(argv[1]) == "--init-plugin") {
        // Child process: initialize exactly one plugin and exit.
        QApplication app(argc, argv);
        return runPluginInitialization(QString::fromUtf8(argv[2]));
    }
    // A custom main (instead of QTEST_MAIN) so the child-process entry point
    // can share this executable.
    QApplication app(argc, argv);
    TestPluginSmoke testObject;
    return QTest::qExec(&testObject, argc, argv);
}

#include "TestPluginSmoke.moc"
