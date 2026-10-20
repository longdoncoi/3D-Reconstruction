#include "PluginHost.h"

#include "IAppContext.h"
#include "IPlugin.h"
#include "PluginDependencyResolver.h"

#include <QDebug>
#include <QDir>
#include <QLibrary>
#include <QPluginLoader>
#include <QStringList>

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
        if (!plugin->initialize(context)) {
            m_errors << QStringLiteral("Plugin %1 failed to initialize; aborting plugin loading.")
                            .arg(plugin->pluginName());
            for (auto it = m_initialized.rbegin(); it != m_initialized.rend(); ++it) {
                (*it)->cleanup();
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
        (*it)->cleanup();
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
    qDebug() << "Loaded plugin:" << plugin->pluginName();
    return plugin;
}

void PluginHost::clearLoaders() {
    for (QPluginLoader *loader : m_loaders) delete loader;
    m_loaders.clear();
}
