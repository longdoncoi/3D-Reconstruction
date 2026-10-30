#include "PluginHost.h"

#include "IAppContext.h"
#include "IPlugin.h"
#include "PluginDependencyResolver.h"

#include <QDebug>
#include <QDir>
#include <QLibrary>
#include <QObject>
#include <QPluginLoader>
#include <QStringList>

#include <exception>
#include <optional>

namespace {

/// Runs one piece of plugin metadata behind the same barrier as the lifecycle
/// hooks.
///
/// Metadata is plugin code too: `pluginName()`, `dependencies()` and the rest
/// are virtuals implemented in the plugin DLL. A `QString` allocation can
/// throw `std::bad_alloc`, an SDK-backed plugin can throw its own type, and a
/// plugin built without `/EHsc` can throw an `int`. Letting one escape has two
/// outcomes, both fatal:
///   * discovery/resolution — the exception leaves `load()` and aborts startup
///     without reporting which plugin caused it;
///   * shutdown — `~PluginHost()` is implicitly `noexcept`, so the same
///     exception becomes `std::terminate()` during application teardown.
/// The barrier turns "this plugin is broken" into "this plugin is rejected",
/// which every caller already handles (drop it + record the reason).
template <typename F>
auto pluginMetadata(const QString &subject, const char *method, QStringList *errors, F &&call)
    -> std::optional<decltype(call())>
{
    try {
        return call();
    } catch (const std::exception &e) {
        *errors << QStringLiteral("Plugin %1: %2 threw: %3")
                       .arg(subject, QString::fromLatin1(method), QString::fromUtf8(e.what()));
    } catch (...) {
        *errors << QStringLiteral("Plugin %1: %2 threw an unknown exception")
                       .arg(subject, QString::fromLatin1(method));
    }
    return std::nullopt;
}

/// The diagnostic name, used from error paths and destructors.
///
/// It must never call back into plugin code that is already known to throw:
/// `initializePlugin()` logs the plugin name from its `catch` block, and a
/// `pluginName()` that throws there would escape the barrier it is reporting
/// on. The same helper keeps `~PluginHost()` total.
QString safePluginName(IPlugin *plugin)
{
    try {
        return plugin->pluginName();
    } catch (...) {
        return QStringLiteral("<unknown plugin>");
    }
}

/// Runs a plugin hook with an exception barrier.
///
/// A plugin is third-party code. `initialize()` and `cleanup()` execute ONCE
/// each, and an exception escaping them reaches the Qt event loop (or
/// `std::terminate()` when it happens on a bare thread) and ABORTS THE WHOLE
/// APPLICATION — before the rollback below, whose entire purpose is to survive
/// a plugin that cannot start, has had a chance to run.
///
/// The barrier turns "the plugin is broken" into "the plugin failed", which is
/// the outcome the rollback already handles: the remaining plugins are cleaned
/// up in reverse order and loading is refused.
bool initializePlugin(IPlugin *plugin, IAppContext *context)
{
    try {
        return plugin->initialize(context);
    } catch (const std::exception &e) {
        qWarning() << "Plugin" << safePluginName(plugin)
                   << "threw during initialize():" << e.what();
    } catch (...) {
        qWarning() << "Plugin" << safePluginName(plugin)
                   << "threw an unknown exception during initialize()";
    }
    return false;
}

/// The cleanup half of the same barrier. A plugin that throws while releasing
/// its widgets has already broken the ownership contract; the process still
/// must not die, and the next plugin's cleanup must still run.
void cleanupPlugin(IPlugin *plugin)
{
    try {
        plugin->cleanup();
    } catch (const std::exception &e) {
        qWarning() << "Plugin" << safePluginName(plugin)
                   << "threw during cleanup():" << e.what();
    } catch (...) {
        qWarning() << "Plugin" << safePluginName(plugin)
                   << "threw an unknown exception during cleanup()";
    }
}

} // namespace

PluginHost::PluginHost() = default;

PluginHost::PluginHost(Loader loader) : m_loader(std::move(loader)) {}

PluginHost::~PluginHost() {
    shutdown();
}

int PluginHost::discover(const QString &directory, const QString &nameFilter) {
    shutdown();
    m_errors.clear();

    QDir dir(directory);
    if (directory.isEmpty() || !dir.exists()) {
        m_errors << QStringLiteral("Plugin directory not found: %1").arg(directory);
        return 0;
    }

    const QStringList files = dir.entryList(QDir::Files, QDir::Name);
    for (const QString &fileName : files) {
        if (!QLibrary::isLibrary(fileName)) continue;
        if (!fileName.contains(nameFilter, Qt::CaseInsensitive)) continue;

        const QString filePath = dir.absoluteFilePath(fileName);
        QString error;
        IPlugin *plugin = m_loader ? m_loader(filePath, &error)
                                   : loadWithQPluginLoader(filePath, &error);
        if (plugin) {
            const auto apiVersion = pluginMetadata(
                fileName, "apiVersion()", &m_errors,
                [plugin] { return plugin->apiVersion(); });
            if (!apiVersion) {
                m_errors << QStringLiteral("Plugin %1: rejected (its apiVersion() threw)")
                                .arg(fileName);
                continue;
            }
            if (*apiVersion != IPlugin::kApiVersion) {
                m_errors << QStringLiteral("Plugin %1: unsupported API version %2 (expected %3)")
                                .arg(fileName)
                                .arg(*apiVersion)
                                .arg(IPlugin::kApiVersion);
                continue;
            }
            const auto abiFingerprint = pluginMetadata(
                fileName, "abiFingerprint()", &m_errors,
                [plugin] { return plugin->abiFingerprint(); });
            if (!abiFingerprint) {
                m_errors << QStringLiteral("Plugin %1: rejected (its abiFingerprint() threw)")
                                .arg(fileName);
                continue;
            }
            if (*abiFingerprint != IPlugin::kAbiFingerprint) {
                m_errors << QStringLiteral(
                                "Plugin %1: incompatible ABI fingerprint 0x%2 (host 0x%3) — "
                                "rebuild it with the same Qt/MSVC/configuration")
                                .arg(fileName)
                                .arg(*abiFingerprint, 0, 16)
                                .arg(IPlugin::kAbiFingerprint, 0, 16);
                continue;
            }
            m_plugins.append(plugin);
        } else {
            m_errors << QStringLiteral("Failed to load plugin %1: %2").arg(fileName, error);
        }
    }
    return m_plugins.size();
}

bool PluginHost::resolveOrder() {
    QList<PluginDependencyResolver::PluginInfo> infos;
    infos.reserve(m_plugins.size());
    for (int i = 0; i < m_plugins.size(); ++i) {
        IPlugin *plugin = m_plugins[i];
        const QString subject = QStringLiteral("index %1").arg(i);

        const auto name = pluginMetadata(
            subject, "pluginName()", &m_errors,
            [plugin] { return plugin->pluginName(); });
        const auto dependencies = pluginMetadata(
            subject, "dependencies()", &m_errors,
            [plugin] { return plugin->dependencies(); });
        const auto phase = pluginMetadata(
            subject, "loadPhase()", &m_errors,
            [plugin] { return plugin->loadPhase(); });
        const auto loadOrder = pluginMetadata(
            subject, "loadOrder()", &m_errors,
            [plugin] { return plugin->loadOrder(); });
        if (!name || !dependencies || !phase || !loadOrder) {
            // Never resolve a partial order: the plugin whose metadata threw
            // is not trustworthy, and the others may depend on it.
            shutdown();
            return false;
        }

        PluginDependencyResolver::PluginInfo info;
        info.name           = *name;
        info.dependencies   = *dependencies;
        info.phase          = *phase;
        info.loadOrder      = *loadOrder;
        info.discoveryIndex = i;
        infos.append(info);
    }

    const PluginDependencyResolver::Result result = PluginDependencyResolver::resolve(infos);
    for (const auto &issue : result.issues) {
        m_errors << QStringLiteral("[plugin-order] %1: %2").arg(issue.plugin, issue.message);
    }
    if (!result.ok) {
        // Never continue with a partial/incorrect order — the resolver already
        // classified this as a wiring bug (cycle, missing dep, duplicate name,
        // unknown phase).
        shutdown();
        return false;
    }

    QList<IPlugin *> ordered;
    ordered.reserve(m_plugins.size());
    for (int index : result.order) ordered.append(m_plugins[index]);
    m_plugins = ordered;
    return true;
}

bool PluginHost::initializeAll(IAppContext *context) {
    for (IPlugin *plugin : m_plugins) {
        if (!initializePlugin(plugin, context)) {
            m_errors << QStringLiteral("Plugin %1 failed to initialize; aborting plugin loading.")
                            .arg(safePluginName(plugin));
            // The failing plugin is cleaned up as well: it may already have
            // installed widgets into the shell, and its DLL is released right
            // after the rollback below.
            cleanupPlugin(plugin);
            for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {
                cleanupPlugin(*it);
            }
            m_initialized.clear();
            m_plugins.clear();
            clearLoaders();
            return false;
        }
        m_initialized.append(plugin);
    }
    return true;
}

void PluginHost::shutdown() {
    for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {
        qDebug() << "Cleaning up plugin:" << safePluginName(*it);
        cleanupPlugin(*it);
    }
    m_initialized.clear();
    m_plugins.clear();
    clearLoaders();
}

QList<IPlugin *> PluginHost::plugins() const {
    return m_plugins;
}

bool PluginHost::isEmpty() const {
    return m_plugins.isEmpty();
}

QStringList PluginHost::takeErrors() {
    const QStringList errors = m_errors;
    m_errors.clear();
    return errors;
}

IPlugin *PluginHost::loadWithQPluginLoader(const QString &filePath, QString *error) {
    auto *loader = new QPluginLoader(filePath);
    QObject *object = loader->instance();
    if (!object) {
        if (error) *error = loader->errorString();
        delete loader;
        return nullptr;
    }
    auto *plugin = qobject_cast<IPlugin *>(object);
    if (!plugin) {
        if (error) *error = QStringLiteral("library does not implement IPlugin");
        delete loader;
        return nullptr;
    }
    m_loaders.push_back(loader);
    m_pluginRoots.push_back(object);
    // The root component has no parent: it is destroyed by the loader's
    // unload(), not by the loader's destructor. The counter makes the
    // release observable — a plugin whose root survives the shutdown keeps
    // its whole QObject tree alive.
    auto counter = std::make_shared<int>(0);
    QObject::connect(object, &QObject::destroyed, [counter]() { ++*counter; });
    m_releaseCounters.push_back(counter);
    qDebug() << "Loaded plugin:" << safePluginName(plugin);
    return plugin;
}

void PluginHost::clearLoaders() {
    for (QPluginLoader *loader : m_loaders) {
        // unload() destroys the plugin's root component, so the plugin's
        // QObject tree goes with it. The plugin's cleanup() must already have
        // run: the shell outlives the plugin, and an object the plugin
        // installed there would stay behind as dead UI.
        //
        // The LIBRARY stays mapped for the rest of the session — Qt's
        // QPluginLoader pins it with QLibrary::PreventUnloadHint and the
        // hints can only be OR-ed, never cleared. That is a Qt constraint,
        // not a defect of the host, and it is why the widget-ownership
        // contract (not the unload) is the one that matters.
        if (!loader->unload()) {
            m_errors << QStringLiteral("Plugin %1: the plugin object could not be released (%2)")
                            .arg(loader->fileName(), loader->errorString());
        }
        delete loader;
    }
    m_loaders.clear();
    m_pluginRoots.clear();
}
