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

namespace {

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
        qWarning() << "Plugin" << plugin->pluginName()
                   << "threw during initialize():" << e.what();
    } catch (...) {
        qWarning() << "Plugin" << plugin->pluginName()
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
        qWarning() << "Plugin" << plugin->pluginName()
                   << "threw during cleanup():" << e.what();
    } catch (...) {
        qWarning() << "Plugin" << plugin->pluginName()
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
            if (plugin->apiVersion() != IPlugin::kApiVersion) {
                m_errors << QStringLiteral("Plugin %1: unsupported API version %2 (expected %3)")
                                .arg(fileName)
                                .arg(plugin->apiVersion())
                                .arg(IPlugin::kApiVersion);
                continue;
            }
            if (plugin->abiFingerprint() != IPlugin::kAbiFingerprint) {
                m_errors << QStringLiteral(
                                "Plugin %1: incompatible ABI fingerprint 0x%2 (host 0x%3) — "
                                "rebuild it with the same Qt/MSVC/configuration")
                                .arg(fileName)
                                .arg(plugin->abiFingerprint(), 0, 16)
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
        PluginDependencyResolver::PluginInfo info;
        info.name           = m_plugins[i]->pluginName();
        info.dependencies   = m_plugins[i]->dependencies();
        info.phase          = m_plugins[i]->loadPhase();
        info.loadOrder      = m_plugins[i]->loadOrder();
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
                            .arg(plugin->pluginName());
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
        qDebug() << "Cleaning up plugin:" << (*it)->pluginName();
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
    qDebug() << "Loaded plugin:" << plugin->pluginName();
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
