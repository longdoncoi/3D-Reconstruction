#include <QtTest>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMap>
#include <QTemporaryDir>

#include "IPlugin.h"
#include "PluginHost.h"

namespace {

/// In-memory IPlugin double: records lifecycle calls into a shared log.
class FakePlugin : public IPlugin {
public:
    FakePlugin(QString name, QStringList *log) : m_name(std::move(name)), m_log(log) {}

    QString pluginName() const override { return m_name; }
    bool initialize(IAppContext * /*context*/) override {
        m_log->append("init:" + m_name);
        return initializeSucceeds;
    }
    void cleanup() override { m_log->append("cleanup:" + m_name); }
    int loadOrder() const override { return order; }
    int apiVersion() const override { return version; }
    quint64 abiFingerprint() const override { return abi; }
    QStringList dependencies() const override { return deps; }
    QString loadPhase() const override { return phase; }

    bool initializeSucceeds = true;
    int order = 100;
    int version = IPlugin::kApiVersion;
    quint64 abi = IPlugin::kAbiFingerprint;
    QStringList deps;
    QString phase = QStringLiteral("default");

private:
    QString m_name;
    QStringList *m_log;
};

} // namespace

/**
 * @brief PluginHost lifecycle contract.
 *
 * The host must not depend on real DLLs to be testable: every test injects a
 * loader that maps a file name to a FakePlugin, while the plugin directory is
 * a real (temporary) folder so discovery/filtering is exercised for real.
 */
class TestPluginHost : public QObject {
    Q_OBJECT

private slots:
    void discoveryFiltersAndLoads();
    void resolveOrderHonoursDependenciesAndPhases();
    void resolutionFailureDropsAllPlugins();
    void initializeFailureRollsBackEarlierPlugins();
    void shutdownCleansUpInReverseInitOrder();
    void missingDirectoryIsReported();
    void apiVersionMismatchIsRejected();
    void abiFingerprintMismatchIsRejected();
    void loaderFailureIsReported();

private:
    static bool writeFile(const QString &path) {
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) return false;
        f.write("stub");
        return true;
    }
};

void TestPluginHost::discoveryFiltersAndLoads()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin alpha("alpha", &log);
    FakePlugin beta("beta", &log);
    const QMap<QString, FakePlugin *> byFile{{"AlphaPlugin.dll", &alpha},
                                             {"BetaPlugin.dll", &beta}};

    QVERIFY(writeFile(pluginsDir.filePath("AlphaPlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("BetaPlugin.dll")));
    // Passes the name filter ("Plugin") but is not a loadable library: the
    // QLibrary::isLibrary gate must reject it before the loader is probed.
    QVERIFY(writeFile(pluginsDir.filePath("NotALibraryPlugin.txt")));
    // Must never be probed by the loader: not a library / no "Plugin" in name.
    QVERIFY(writeFile(pluginsDir.filePath("notes.txt")));
    QVERIFY(writeFile(pluginsDir.filePath("AppCore.dll")));

    PluginHost host([&byFile](const QString &path, QString *error) -> IPlugin * {
        const QString name = QFileInfo(path).fileName();
        if (!byFile.contains(name)) {
            if (error) *error = "unexpected probe: " + name;
            return nullptr;
        }
        return byFile.value(name);
    });

    QCOMPARE(host.discover(pluginsDir.path()), 2);
    QVERIFY2(host.errors().isEmpty(),
             qPrintable(host.errors().join(QStringLiteral(" | "))));
    QCOMPARE(host.plugins().size(), 2);
    QVERIFY(!host.isEmpty());
    QVERIFY(log.isEmpty()); // discovery must not initialize anything
}

void TestPluginHost::resolveOrderHonoursDependenciesAndPhases()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin alpha("alpha", &log);           // discovery: first by file name
    FakePlugin beta("beta", &log);
    beta.deps = {"alpha"};
    FakePlugin gamma("gamma", &log);
    gamma.phase = QStringLiteral("kernel");    // phase hint wins over discovery order
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha},
                                             {"BPlugin.dll", &beta},
                                             {"CPlugin.dll", &gamma}};

    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("BPlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("CPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 3);
    QVERIFY(host.resolveOrder());
    QVERIFY2(host.errors().isEmpty(),
             qPrintable(host.errors().join(QStringLiteral(" | "))));

    QStringList names;
    for (IPlugin *plugin : host.plugins()) names << plugin->pluginName();
    QCOMPARE(names, QStringList({"gamma", "alpha", "beta"}));
}

void TestPluginHost::resolutionFailureDropsAllPlugins()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin alpha("alpha", &log);
    alpha.deps = {"ghost"}; // missing dependency — wiring bug, must fail fast
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha}};
    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 1);
    QVERIFY(!host.resolveOrder());
    QVERIFY(host.isEmpty()); // never continue with a partial order
    const QStringList errors = host.takeErrors();
    QVERIFY(!errors.isEmpty());
    QVERIFY(errors.first().startsWith(QStringLiteral("[plugin-order]")));
    QVERIFY(host.takeErrors().isEmpty()); // taking clears
}

void TestPluginHost::initializeFailureRollsBackEarlierPlugins()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin alpha("alpha", &log);
    alpha.order = 10;
    FakePlugin beta("beta", &log);
    beta.order = 20;
    beta.deps = {"alpha"};
    beta.initializeSucceeds = false;
    FakePlugin gamma("gamma", &log);
    gamma.order = 30;
    gamma.deps = {"beta"};
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha},
                                             {"BPlugin.dll", &beta},
                                             {"CPlugin.dll", &gamma}};

    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("BPlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("CPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 3);
    QVERIFY(host.resolveOrder());

    QVERIFY(!host.initializeAll(nullptr));
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "cleanup:alpha"}));
    QVERIFY(host.isEmpty());
    QVERIFY(host.errors().last().contains(QStringLiteral("failed to initialize")));

    // shutdown() after a failed init must not clean anything up again.
    host.shutdown();
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "cleanup:alpha"}));
}

void TestPluginHost::shutdownCleansUpInReverseInitOrder()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin alpha("alpha", &log);
    alpha.order = 10;
    FakePlugin beta("beta", &log);
    beta.order = 20;
    beta.deps = {"alpha"};
    FakePlugin gamma("gamma", &log);
    gamma.order = 30;
    gamma.deps = {"beta"};
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha},
                                             {"BPlugin.dll", &beta},
                                             {"CPlugin.dll", &gamma}};

    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("BPlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("CPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 3);
    QVERIFY(host.resolveOrder());
    QVERIFY(host.initializeAll(nullptr));
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "init:gamma"}));

    host.shutdown();
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "init:gamma",
                               "cleanup:gamma", "cleanup:beta", "cleanup:alpha"}));
    QVERIFY(host.isEmpty());

    // Idempotent: a second shutdown (e.g. dtor after closeEvent) is a no-op.
    host.shutdown();
    QCOMPARE(log.size(), 6);
}

void TestPluginHost::missingDirectoryIsReported()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    PluginHost host([](const QString &, QString *error) -> IPlugin * {
        if (error) *error = "loader must not be called for a missing directory";
        return nullptr;
    });

    QCOMPARE(host.discover(dir.filePath("does-not-exist")), 0);
    QCOMPARE(host.errors().size(), 1);
    QVERIFY(host.errors().first().contains(QStringLiteral("not found")));
    QVERIFY(!host.takeErrors().isEmpty());
    QVERIFY(host.errors().isEmpty());
}

void TestPluginHost::apiVersionMismatchIsRejected()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin stale("stale", &log);
    stale.version = IPlugin::kApiVersion + 1; // built against a newer contract
    const QMap<QString, FakePlugin *> byFile{{"StalePlugin.dll", &stale}};
    QVERIFY(writeFile(pluginsDir.filePath("StalePlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 0);
    QVERIFY(host.isEmpty());
    QCOMPARE(host.errors().size(), 1);
    QVERIFY(host.errors().first().contains(QStringLiteral("API version")));
    QVERIFY(log.isEmpty());
}

void TestPluginHost::abiFingerprintMismatchIsRejected()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin stale("stale-abi", &log);
    stale.abi = IPlugin::kAbiFingerprint ^ 0x1ULL; // different toolchain/runtime
    const QMap<QString, FakePlugin *> byFile{{"StaleAbiPlugin.dll", &stale}};
    QVERIFY(writeFile(pluginsDir.filePath("StaleAbiPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 0);
    QVERIFY(host.isEmpty());
    QCOMPARE(host.errors().size(), 1);
    QVERIFY(host.errors().first().contains(QStringLiteral("ABI fingerprint")));
    QVERIFY(log.isEmpty());
}

void TestPluginHost::loaderFailureIsReported()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));
    QVERIFY(writeFile(pluginsDir.filePath("BrokenPlugin.dll")));

    PluginHost host([](const QString & /*path*/, QString *error) -> IPlugin * {
        if (error) *error = QStringLiteral("simulated load failure");
        return nullptr;
    });

    // A file that passes the name + library filters but fails to load must be
    // reported: silently dropping it would let a plugin disappear without a
    // trace and leave its dependencies unresolved later.
    QCOMPARE(host.discover(pluginsDir.path()), 0);
    QVERIFY(host.isEmpty());
    QCOMPARE(host.errors().size(), 1);
    QVERIFY2(host.errors().first().contains(QStringLiteral("simulated load failure")),
             qPrintable(host.errors().join(QStringLiteral(" | "))));
}

QTEST_MAIN(TestPluginHost)
#include "TestPluginHost.moc"
