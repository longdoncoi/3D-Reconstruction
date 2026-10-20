#ifndef IPLUGIN_H
#define IPLUGIN_H

#include <QtGlobal>
#include <QtPlugin>
#include <QString>

namespace PluginAbiDetail {

#if defined(_MSC_VER)
inline constexpr quint64 compiler = static_cast<quint64>(_MSC_VER);
#else
inline constexpr quint64 compiler = 0;
#endif

#if defined(_ITERATOR_DEBUG_LEVEL)
inline constexpr quint64 iteratorDebug = static_cast<quint64>(_ITERATOR_DEBUG_LEVEL);
#else
inline constexpr quint64 iteratorDebug = 0;
#endif

#ifdef QT_NO_DEBUG
inline constexpr quint64 debugBuild = 0;
#else
inline constexpr quint64 debugBuild = 1;
#endif

} // namespace PluginAbiDetail

class IAppContext;

/**
 * @brief Plugin lifecycle contract resolved by PluginDependencyResolver
 *        BEFORE any initialize() call.
 *
 * Ordering rules (hard edge always wins over hints):
 *  1. dependencies() — names of plugins that must initialize first.
 *  2. loadPhase()    — coarse grouping hint; canonical values (low → high):
 *                      "kernel" < "platform" < "default" < "ui" < "late".
 *  3. loadOrder()    — fine-grained tie-break inside one phase (lower first).
 *  4. Discovery order of the plugin directory (stable tie-break).
 *
 * Missing dependencies, duplicate plugin names, unknown phases and dependency
 * cycles abort plugin loading with a concrete error — they are wiring bugs,
 * not runtime conditions (same fail-fast stance as ServiceRegistry).
 */
class IPlugin {
public:
    /// Version of the IPlugin/plugin-host contract. Bump it whenever the
    /// virtual surface below or a plugin-facing service interface changes;
    /// PluginHost then refuses to load plugins built against a different
    /// version instead of crashing on a stale ABI.
    ///
    /// v3: IShellService::getProgressDialog() returns IProgressDialog (the
    ///     concrete CustomProgressDialog no longer derives QDialog).
    static constexpr int kApiVersion = 3;

    /**
     * Toolchain/ABI fingerprint in addition to kApiVersion. The virtual
     * surface can stay identical while Qt, the MSVC runtime, the iterator
     * debugging level or the build configuration differ; mixing those across
     * the exe ↔ plugin DLL boundary is undefined behaviour (especially for
     * std::vector/std::shared_ptr in the interface). The host rejects a
     * plugin built with a different fingerprint instead of loading it.
     */
    static constexpr quint64 kAbiFingerprint =
        (static_cast<quint64>(QT_VERSION) << 32)
        | (PluginAbiDetail::compiler << 16)
        | (PluginAbiDetail::iteratorDebug << 8)
        | PluginAbiDetail::debugBuild;

    virtual ~IPlugin() = default;
    virtual QString pluginName() const = 0;
    [[nodiscard]] virtual bool initialize(IAppContext* context) = 0;
    virtual void cleanup() = 0;

    /// Contract version implemented by this plugin (defaults to kApiVersion).
    virtual int apiVersion() const { return kApiVersion; }

    /// Toolchain fingerprint implemented by this plugin (defaults to the
    /// value compiled into the host header).
    virtual quint64 abiFingerprint() const { return kAbiFingerprint; }

    /// Fine-grained init-order hint inside one phase (lower first).
    virtual int  loadOrder() const { return 100; }

    /// Names of plugins that MUST be initialized before this one.
    virtual QStringList dependencies() const { return {}; }

    /// Coarse init-order hint; see canonical phase list above.
    virtual QString loadPhase() const { return QStringLiteral("default"); }
};

#define IPlugin_iid "com.3dreconstruction.IPlugin"
Q_DECLARE_INTERFACE(IPlugin, IPlugin_iid)

#endif // IPLUGIN_H
