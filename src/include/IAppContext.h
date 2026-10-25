#ifndef IAPPCONTEXT_H
#define IAPPCONTEXT_H

#include <QString>
#include "ServiceRegistry.h"

// Only forward declarations — the context interface must not drag concrete
// UI/GUI types (QMainWindow, VTK, dialog widgets) into every consumer. Files
// that dereference these types include their headers themselves.
class SignalBus;
class IShellService;

class ISceneService;
class IViewerService;
class ISettingsService;

/**
 * @brief Kernel contract handed to plugins at initialize().
 *
 * Deliberately narrow: it exposes the application services (directly and via
 * the ServiceRegistry) and the SignalBus. UI-shell concerns (named menus, tab
 * panels, the shared progress dialog) live in IShellService — resolve it via
 * shell() or the registry. Plugins never receive the concrete QMainWindow:
 * every shell interaction goes through IShellService.
 *
 * Access paths — two-tier contract, documented on purpose (do not add a
 * third): the direct getters (scene/viewer/settings/shell/signalBus) are for
 * the core services every plugin may rely on; services()->get<T>() is for
 * optional / late-registered services (IMailService, IReconstructionService,
 * IAIService, IUserManager, …).
 *
 * services() is READ-ONLY by design. Registration belongs to the composition
 * root (AppShell owns the MutableServiceRegistry from src/core); a plugin
 * that could unregister a service and install a replacement would defeat the
 * "first registration wins" rule and the service-locator contract.
 *
 * translate() stays as the single i18n convenience for UI code; it delegates
 * to ILanguageService, which remains the canonical translation interface.
 */
class IAppContext {
public:
    virtual ~IAppContext() = default;

    virtual ISceneService*    scene()    = 0;
    virtual IViewerService*   viewer()   = 0;
    virtual ISettingsService* settings() = 0;
    virtual SignalBus*        signalBus()  = 0;

    /** UI-shell contract (menus, tab panels, shared progress dialog). */
    virtual IShellService*    shell() = 0;

    /**
     * @brief Read-only ServiceRegistry for plugin look-ups.
     *
     * Ví dụ:
     *   auto* svc = ctx->services()->get<IReconstructionService>();
     *
     * The returned view cannot register, unregister or clear services: those
     * are composition-root operations (MutableServiceRegistry in src/core).
     */
    virtual const ServiceRegistry* services() = 0;

    virtual QString translate(const QString& key) = 0;
    virtual void    setLanguage(const QString& lang) = 0;
};

#endif // IAPPCONTEXT_H
