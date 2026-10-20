#include <QtTest>

#include <QDir>
#include <QFile>
#include <QLibrary>
#include <QSettings>
#include <QTemporaryDir>

#include "PluginDependencyResolver.h"
#include "PluginHost.h"
#include "UserSettingsStore.h"

/**
 * @brief The remaining branches of the guard layer.
 *
 * Everything here is a decision that must fail loudly:
 *
 * - the plugin host's QPluginLoader failure paths (a file that is not a
 *   library, a library that does not implement IPlugin),
 * - the dependency resolver's wiring diagnostics (an empty plugin name, a
 *   duplicated name, a dependency that is not loaded),
 * - the per-user settings fallback to a legacy plaintext preference.
 */
class TestGuardBranches : public QObject {
    Q_OBJECT

private slots:
    void pluginHostReportsAFileThatIsNotALibrary();
    void pluginHostReportsALibraryWithoutThePluginInterface();
    void resolverReportsAnEmptyPluginName();
    void resolverReportsADuplicatedPluginName();
    void resolverReportsAMissingDependency();
    void userSettingsFallBackToALegacyPlaintextPreference();
};

void TestGuardBranches::pluginHostReportsAFileThatIsNotALibrary()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    // A file with a library name that is not a library at all.
    QFile broken(dir.filePath(QStringLiteral("BrokenPlugin.dll")));
    QVERIFY(broken.open(QIODevice::WriteOnly));
    broken.write("this is not a PE image");
    broken.close();

    PluginHost host; // the default loader: QPluginLoader
    host.discover(dir.path());

    QVERIFY2(!host.errors().isEmpty(), "a broken plugin file must be reported");
    QVERIFY2(host.errors().first().contains(QStringLiteral("BrokenPlugin.dll")),
             qPrintable(host.errors().join(QStringLiteral(" | "))));
    QVERIFY(host.isEmpty());
}

void TestGuardBranches::pluginHostReportsALibraryWithoutThePluginInterface()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    // A real Qt plugin (an image format handler) — it loads, but it does not
    // implement the application's IPlugin interface.
    QString source;
    for (const QString &libraryPath : QCoreApplication::libraryPaths()) {
        const QString candidate = libraryPath + QStringLiteral("/imageformats/qjpeg.dll");
        if (QFile::exists(candidate)) {
            source = candidate;
            break;
        }
    }
    if (source.isEmpty()) {
        QSKIP("no Qt image-format plugin found in the library paths");
    }
    QVERIFY(QFile::copy(source, dir.filePath(QStringLiteral("NotAPlugin.dll"))));

    PluginHost host;
    host.discover(dir.path());

    QVERIFY2(!host.errors().isEmpty(), "a library without IPlugin must be reported");
    QVERIFY(host.isEmpty());
}

void TestGuardBranches::resolverReportsAnEmptyPluginName()
{
    PluginDependencyResolver::PluginInfo nameless;
    nameless.name = QString();
    nameless.discoveryIndex = 0;

    const auto result = PluginDependencyResolver::resolve({nameless});
    QVERIFY(!result.ok);
    QCOMPARE(result.issues.size(), 1);
    QVERIFY(result.issues.first().message.contains(QStringLiteral("empty name")));
}

void TestGuardBranches::resolverReportsADuplicatedPluginName()
{
    PluginDependencyResolver::PluginInfo first;
    first.name = QStringLiteral("Same");
    first.discoveryIndex = 0;
    PluginDependencyResolver::PluginInfo second;
    second.name = QStringLiteral("Same");
    second.discoveryIndex = 1;

    const auto result = PluginDependencyResolver::resolve({first, second});
    QVERIFY(!result.ok);
    QVERIFY2(result.issues.first().message.contains(QStringLiteral("duplicate plugin name")),
             qPrintable(result.issues.first().message));
}

void TestGuardBranches::resolverReportsAMissingDependency()
{
    PluginDependencyResolver::PluginInfo plugin;
    plugin.name = QStringLiteral("Consumer");
    plugin.dependencies = QStringList{QStringLiteral("Missing")};
    plugin.discoveryIndex = 0;

    const auto result = PluginDependencyResolver::resolve({plugin});
    QVERIFY(!result.ok);
    QVERIFY2(result.issues.first().message.contains(QStringLiteral("Missing")),
             qPrintable(result.issues.first().message));
}

void TestGuardBranches::userSettingsFallBackToALegacyPlaintextPreference()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QSettings settings(dir.filePath(QStringLiteral("settings.ini")), QSettings::IniFormat);
    // A preference written before the secret storage existed: it lives under
    // "pref_<key>" and is not encrypted.
    settings.setValue(QStringLiteral("User_alice/pref_theme"), QStringLiteral("dark"));
    settings.sync();

    UserSettingsStore store;
    store.attachSettings(&settings);

    QCOMPARE(store.getUserSecret(QStringLiteral("alice"), QStringLiteral("theme")),
             QStringLiteral("dark"));
    // An unknown key still returns the caller's default.
    QCOMPARE(store.getUserSecret(QStringLiteral("alice"), QStringLiteral("nope"),
                                 QStringLiteral("fallback")),
             QStringLiteral("fallback"));
}

QTEST_GUILESS_MAIN(TestGuardBranches)
#include "TestGuardBranches.moc"
