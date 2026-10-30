#ifndef PLUGINHOST_H
#define PLUGINHOST_H

#include <QString>
#include <QStringList>
#include <functional>
#include <memory>
#include <vector>
#include "Global.h"

class IAppContext;
class IPlugin;
class QObject;
class QPluginLoader;

/**
 * @brief Owns the plugin load → resolve → initialize → cleanup lifecycle.
 *
 * Extracted from the exe shell (MainWindow) so the contract is compiled into
 * AppCore and unit-testable with an injected loader (tests never build a real
 * DLL). Guarantees:
 *  - discovery only probes library files whose name matches a filter,
 *  - dependency/phase ordering is delegated to PluginDependencyResolver and a
 *    resolution failure drops every plugin (never load a partial order),
 *  - initialize() runs in resolved order with rollback of the already
 *    initialized plugins on the first failure,
 *  - cleanup() runs in reverse order of successful initialization, exactly
 *    once per plugin.
 *
 * Every call into plugin code — lifecycle hooks AND metadata virtuals
 * (`pluginName()`, `apiVersion()`, `abiFingerprint()`, `dependencies()`,
 * `loadPhase()`, `loadOrder()`) — goes through an exception barrier. Metadata
 * throws are not hypothetical: a QString allocation or an SDK exception in a
 * plugin virtual would otherwise escape discovery, or reach `std::terminate()`
 * through the implicitly-noexcept destructor. A throwing plugin is rejected
 * with an error instead.
 *
 * The shell keeps the two responsibilities that need the concrete window:
 * picking the plugin directory and registering the deferred services before
 * initializeAll() is called.
 */
class APP_EXPORT PluginHost {
public:
    /// Loads one library and returns its IPlugin instance (nullptr + *error on
    /// failure). Defaults to QPluginLoader; tests inject a fake.
    using Loader = std::function<IPlugin *(const QString &filePath, QString *error)>;

    /// Uses QPluginLoader; the loaders stay alive for the host's lifetime.
    PluginHost();
    /// Test seam: custom loader — no QPluginLoader instance is created.
    explicit PluginHost(Loader loader);
    ~PluginHost();

    PluginHost(const PluginHost &) = delete;
    PluginHost &operator=(const PluginHost &) = delete;

    /// (Re)loads plugins from @p directory. Libraries whose file name does not
    /// contain @p nameFilter (case-insensitive) are ignored, so Qt/vcpkg
    /// dependencies are never probed. Returns the number of loaded plugins;
    /// failures are reported through errors().
    int discover(const QString &directory,
                 const QString &nameFilter = QStringLiteral("Plugin"));

    /// Resolves initialize() order (dependencies > phase > loadOrder >
    /// discovery order). On failure errors() carries the concrete issue list
    /// and every plugin is dropped — the caller must not continue loading.
    bool resolveOrder();

    /// initialize() in resolved order. On the first failure every plugin that
    /// already initialized is cleaned up (reverse order) and the host empties
    /// itself; returns false so the caller can abort the application.
    bool initializeAll(IAppContext *context);

    /// cleanup() in reverse order of successful initialization, then drops the
    /// plugins and releases them. Safe to call repeatedly or early
    /// (before discover/initialize).
    ///
    /// The release destroys each plugin's root component, so the plugin's
    /// QObject tree goes with it. The *library* itself stays mapped for the
    /// rest of the session: QPluginLoader pins it with
    /// QLibrary::PreventUnloadHint, and Qt only ever ORs the load hints, so
    /// the hint cannot be cleared. A plugin object that outlives cleanup()
    /// is therefore dead UI rather than an immediate crash — but the shell
    /// outlives the plugin, so the contract still is: **cleanup() must
    /// release every object the plugin installed into the shell.** That is
    /// verified by TestPluginSmoke.
    void shutdown();

    QList<IPlugin *> plugins() const;
    bool isEmpty() const;

    /// The number of plugin root components that were released (destroyed) so
    /// far. A plugin whose root is still alive after the shutdown keeps its
    /// whole QObject tree — and every object it installed in the shell.
    int releasedPluginCount() const
    {
        int count = 0;
        for (const auto &counter : m_releaseCounters) count += *counter;
        return count;
    }

    /// Errors accumulated so far (discovery, resolution, initialization).
    QStringList errors() const { return m_errors; }
    /// Moves the errors out (clearing them) — for shells that log per phase.
    QStringList takeErrors();

private:
    IPlugin *loadWithQPluginLoader(const QString &filePath, QString *error);
    void clearLoaders();

    Loader m_loader;                        ///< empty → QPluginLoader-based default
    std::vector<QPluginLoader *> m_loaders; ///< default loader's instances
    std::vector<QObject *> m_pluginRoots;   ///< the root component of each loader
    /// One counter per loaded plugin, incremented when its root component is
    /// destroyed. Shared with the destroyed() handler so the handler stays
    /// valid even if the host is gone first.
    std::vector<std::shared_ptr<int>> m_releaseCounters;
    QList<IPlugin *> m_plugins;             ///< discovered, then resolved order
    QList<IPlugin *> m_initialized;         ///< successfully initialized (init order)
    QStringList m_errors;
};

#endif // PLUGINHOST_H
