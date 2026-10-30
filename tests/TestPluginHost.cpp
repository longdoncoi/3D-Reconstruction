#include <QtTest>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMap>
#include <QTemporaryDir>

#include <stdexcept>

#include "IPlugin.h"
#include "PluginHost.h"

namespace {

/// In-memory IPlugin double: records lifecycle calls into a shared log.
class FakePlugin : public IPlugin {
public:
    FakePlugin(QString name, QStringList *log) : m_name(std::move(name)), m_log(log) {}

    QString pluginName() const override {
        if (metadataThrowsUnknown) throw 42;   // only catch (...) can contain it
        if (pluginNameThrows)
            throw std::runtime_error("the plugin exploded in pluginName()");
        return m_name;
    }
    bool initialize(IAppContext * /*context*/) override {
        m_log->append("init:" + m_name);
        if (initializeThrowsUnknown)
            throw 42;   // a non-std type: only catch (...) can contain it
        if (initializeThrows) {
            // The barrier logs the plugin name from its catch block; a plugin
            // that throws there too must not escape the barrier it reports on.
            if (pluginNameThrowsOnInitializeFailure) pluginNameThrows = true;
            throw std::runtime_error("the plugin exploded during initialize");
        }
        return initializeSucceeds;
    }
    void cleanup() override {
        m_log->append("cleanup:" + m_name);
        if (cleanupThrowsUnknown)
            throw 42;   // a non-std type reaches catch (...) only
        if (cleanupThrows)
            throw std::runtime_error("the plugin exploded during cleanup");
    }
    int loadOrder() const override {
        if (loadOrderThrows) throw std::runtime_error("the plugin exploded in loadOrder()");
        return order;
    }
    int apiVersion() const override {
        if (apiVersionThrows) throw std::runtime_error("the plugin exploded in apiVersion()");
        return version;
    }
    quint64 abiFingerprint() const override {
        if (abiFingerprintThrows)
            throw std::runtime_error("the plugin exploded in abiFingerprint()");
        return abi;
    }
    QStringList dependencies() const override {
        if (dependenciesThrows)
            throw std::runtime_error("the plugin exploded in dependencies()");
        return deps;
    }
    QString loadPhase() const override {
        if (loadPhaseThrows) throw std::runtime_error("the plugin exploded in loadPhase()");
        return phase;
    }

    bool initializeSucceeds = true;
    bool initializeThrows = false;
    bool initializeThrowsUnknown = false;
    bool cleanupThrows = false;
    bool cleanupThrowsUnknown = false;
    bool pluginNameThrows = false;
    bool pluginNameThrowsOnInitializeFailure = false;
    bool apiVersionThrows = false;
    bool abiFingerprintThrows = false;
    bool dependenciesThrows = false;
    bool loadPhaseThrows = false;
    bool loadOrderThrows = false;
    bool metadataThrowsUnknown = false;
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
    void initializeExceptionRollsBackInsteadOfAborting();
    void aNonStandardExceptionIsContainedToo();
    void metadataExceptionRejectsThePlugin_data();
    void metadataExceptionRejectsThePlugin();
    void aNonStandardMetadataExceptionIsContained();
    void aThrowingPluginNameDoesNotEscapeShutdown();
    void aThrowingPluginNameDoesNotEscapeTheInitializeBarrier();
    void cleanupExceptionDoesNotStopTheRemainingCleanups();
    void shutdownCleansUpInReverseInitOrder();
    void missingDirectoryIsReported();
    void apiVersionMismatchIsRejected();
    void abiFingerprintMismatchIsRejected();
    void abiFingerprintCarriesTheInterfaceHash();
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
    // The failing plugin is cleaned up as well: it may already have
    // installed objects into the shell, and its DLL is released right after
    // the rollback. The order is LIFO: the failing one first, then the
    // already initialized ones.
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "cleanup:beta", "cleanup:alpha"}));
    QVERIFY(host.isEmpty());
    QVERIFY(host.errors().last().contains(QStringLiteral("failed to initialize")));

    // shutdown() after a failed init must not clean anything up again.
    host.shutdown();
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "cleanup:beta", "cleanup:alpha"}));
}

void TestPluginHost::initializeExceptionRollsBackInsteadOfAborting()
{
    // A plugin is third-party code: initialize() may throw. Before the barrier
    // the exception escaped into the event loop and ABORTED THE PROCESS, so the
    // rollback — the whole point of a failed init — never ran.
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
    beta.initializeThrows = true;
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha},
                                             {"BPlugin.dll", &beta}};
    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("BPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 2);
    QVERIFY(host.resolveOrder());

    // A throwing plugin is a FAILED plugin, not a crash: the failing one is
    // cleaned up (it may have installed widgets before throwing) and the
    // already-initialized ones are rolled back in LIFO order.
    QVERIFY(!host.initializeAll(nullptr));
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "cleanup:beta", "cleanup:alpha"}));
    QVERIFY(host.isEmpty());
    QVERIFY(host.errors().last().contains(QStringLiteral("failed to initialize")));
}

void TestPluginHost::aNonStandardExceptionIsContainedToo()
{
    // `catch (const std::exception&)` is not enough: a plugin built without
    // /EHsc, or one wrapping an SDK that throws an int/HRESULT, throws a type
    // that only `catch (...)` sees. Without that clause the exception leaves
    // the barrier and kills the process anyway.
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
    beta.initializeThrowsUnknown = true;
    beta.cleanupThrowsUnknown = true;   // the rollback cleanup throws too
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha},
                                             {"BPlugin.dll", &beta}};
    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));
    QVERIFY(writeFile(pluginsDir.filePath("BPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 2);
    QVERIFY(host.resolveOrder());

    // Both barriers contain the non-std throw: the run fails as a rejected
    // load, not as a crash, and the earlier plugin is still rolled back.
    QVERIFY(!host.initializeAll(nullptr));
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "cleanup:beta", "cleanup:alpha"}));
    QVERIFY(host.isEmpty());
}

void TestPluginHost::metadataExceptionRejectsThePlugin_data()
{
    QTest::addColumn<QString>("hook");
    QTest::newRow("pluginName") << QStringLiteral("pluginName");
    QTest::newRow("dependencies") << QStringLiteral("dependencies");
    QTest::newRow("loadPhase") << QStringLiteral("loadPhase");
    QTest::newRow("loadOrder") << QStringLiteral("loadOrder");
    QTest::newRow("apiVersion") << QStringLiteral("apiVersion");
    QTest::newRow("abiFingerprint") << QStringLiteral("abiFingerprint");
}

void TestPluginHost::metadataExceptionRejectsThePlugin()
{
    // Metadata is plugin code too. Before the barrier, a throw from any of
    // these virtuals escaped discovery/resolution (startup died with no
    // diagnosis) or, in shutdown, reached std::terminate() through the
    // implicitly-noexcept destructor. The class is pinned per hook, not by one
    // representative: every metadata virtual has a row.
    QFETCH(QString, hook);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin broken("broken", &log);
    if (hook == QLatin1String("pluginName")) broken.pluginNameThrows = true;
    else if (hook == QLatin1String("dependencies")) broken.dependenciesThrows = true;
    else if (hook == QLatin1String("loadPhase")) broken.loadPhaseThrows = true;
    else if (hook == QLatin1String("loadOrder")) broken.loadOrderThrows = true;
    else if (hook == QLatin1String("apiVersion")) broken.apiVersionThrows = true;
    else if (hook == QLatin1String("abiFingerprint")) broken.abiFingerprintThrows = true;
    else QFAIL("unknown hook");

    const QMap<QString, FakePlugin *> byFile{{"BrokenPlugin.dll", &broken}};
    QVERIFY(writeFile(pluginsDir.filePath("BrokenPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    const bool discoveryHook =
        hook == QLatin1String("apiVersion") || hook == QLatin1String("abiFingerprint");
    if (discoveryHook) {
        // The ABI gate sees the throw during discovery: the plugin is rejected
        // before it can be appended to m_plugins.
        QCOMPARE(host.discover(pluginsDir.path()), 0);
        QVERIFY(host.isEmpty());
    } else {
        QCOMPARE(host.discover(pluginsDir.path()), 1);
        QVERIFY(!host.resolveOrder());
        QVERIFY(host.isEmpty());   // never continue with a partial order
    }

    const QStringList errors = host.takeErrors();
    QVERIFY2(!errors.isEmpty(), "a throwing metadata hook must be reported");
    bool mentionsHook = false;
    for (const QString &error : errors) {
        if (error.contains(hook)) {
            mentionsHook = true;
            break;
        }
    }
    QVERIFY2(mentionsHook, qPrintable(errors.join(QStringLiteral(" | "))));
}

void TestPluginHost::aNonStandardMetadataExceptionIsContained()
{
    // `catch (const std::exception&)` alone would miss `throw 42` — the same
    // reason the lifecycle barrier has a catch (...) clause.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin broken("broken", &log);
    broken.pluginNameThrows = true;
    broken.metadataThrowsUnknown = true;
    const QMap<QString, FakePlugin *> byFile{{"BrokenPlugin.dll", &broken}};
    QVERIFY(writeFile(pluginsDir.filePath("BrokenPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 1);
    QVERIFY(!host.resolveOrder());
    QVERIFY(host.isEmpty());
    QVERIFY2(host.errors().join(QStringLiteral(" | ")).contains(
                 QStringLiteral("unknown exception")),
             qPrintable(host.errors().join(QStringLiteral(" | "))));
}

void TestPluginHost::aThrowingPluginNameDoesNotEscapeShutdown()
{
    // ~PluginHost() is implicitly noexcept and calls shutdown(). A pluginName()
    // that throws there used to reach std::terminate() at application exit.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin alpha("alpha", &log);
    const QMap<QString, FakePlugin *> byFile{{"APlugin.dll", &alpha}};
    QVERIFY(writeFile(pluginsDir.filePath("APlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 1);
    QVERIFY(host.resolveOrder());
    QVERIFY(host.initializeAll(nullptr));

    alpha.pluginNameThrows = true;   // only the teardown log can see it now
    host.shutdown();                 // must not terminate the process
    QVERIFY(host.isEmpty());
    QCOMPARE(log, QStringList({"init:alpha", "cleanup:alpha"}));
}

void TestPluginHost::aThrowingPluginNameDoesNotEscapeTheInitializeBarrier()
{
    // The barrier logs the plugin name from its catch block. If that second
    // call also throws, the exception escapes the barrier that is reporting
    // the first one — the failure mode is still std::terminate().
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QDir pluginsDir(dir.filePath("plugins"));
    QVERIFY(QDir().mkpath(pluginsDir.path()));

    QStringList log;
    FakePlugin broken("broken", &log);
    broken.initializeThrows = true;
    broken.pluginNameThrowsOnInitializeFailure = true;
    const QMap<QString, FakePlugin *> byFile{{"BrokenPlugin.dll", &broken}};
    QVERIFY(writeFile(pluginsDir.filePath("BrokenPlugin.dll")));

    PluginHost host([&byFile](const QString &path, QString *) -> IPlugin * {
        return byFile.value(QFileInfo(path).fileName());
    });

    QCOMPARE(host.discover(pluginsDir.path()), 1);
    QVERIFY(host.resolveOrder());
    QVERIFY(!host.initializeAll(nullptr));
    QVERIFY(host.isEmpty());
    QVERIFY2(host.errors().last().contains(QStringLiteral("<unknown plugin>")),
             qPrintable(host.errors().join(QStringLiteral(" | "))));
}

void TestPluginHost::cleanupExceptionDoesNotStopTheRemainingCleanups()
{
    // cleanup() releases widgets the shell owns. A plugin that throws halfway
    // leaves stale UI behind (its own bug), but the process must survive and
    // every other plugin must still be cleaned up.
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
    beta.cleanupThrows = true;
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

    host.shutdown();
    QCOMPARE(log, QStringList({"init:alpha", "init:beta", "init:gamma",
                               "cleanup:gamma", "cleanup:beta", "cleanup:alpha"}));
    QVERIFY(host.isEmpty());
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

void TestPluginHost::abiFingerprintCarriesTheInterfaceHash()
{
    // The fingerprint the host compares must incorporate the content hash of
    // the plugin-facing interface surface: that is what makes a plugin built
    // against different I*.h fail the gate even when nobody bumped
    // kApiVersion. TestInterfaceAbi guards the other half of the contract
    // (the hash is derived from the CURRENT headers and reacts to a change).
    QVERIFY(IPlugin::kInterfaceHash != 0);
    QVERIFY2(IPlugin::kSdkFingerprint != 0,
             "the SDK fingerprint is empty — a VTK/OpenCV/ONNX bump would be "
             "invisible to the ABI gate");
    QCOMPARE(IPlugin::kAbiFingerprint,
             (IPlugin::kToolchainFingerprint ^ IPlugin::kSdkFingerprint)
                 ^ (IPlugin::kInterfaceHash * 0x9E3779B97F4A7C15ULL));
    // A different interface surface must produce a different fingerprint.
    QVERIFY((IPlugin::kToolchainFingerprint ^ IPlugin::kSdkFingerprint
             ^ ((IPlugin::kInterfaceHash ^ 0x1ULL) * 0x9E3779B97F4A7C15ULL))
            != IPlugin::kAbiFingerprint);
    // A different SDK build must produce a different fingerprint too: VTK /
    // OpenCV / ONNX Runtime types cross the plugin boundary (ISceneService,
    // IAIService), so a SDK bump that changes their layout is an ABI bump.
    QVERIFY((IPlugin::kToolchainFingerprint ^ (IPlugin::kSdkFingerprint ^ 0x1ULL)
             ^ (IPlugin::kInterfaceHash * 0x9E3779B97F4A7C15ULL))
            != IPlugin::kAbiFingerprint);
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
