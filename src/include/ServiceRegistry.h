#ifndef SERVICEREGISTRY_H
#define SERVICEREGISTRY_H

#include <QHash>
#include <QString>
#include <QDebug>

#include <concepts>
#include <shared_mutex>
#include <type_traits>

/**
 * @brief Type-safe service registry (Service Locator pattern).
 *
 * Every registrable interface declares a deterministic compile-time key:
 *
 *     class IReconstructionService {
 *     public:
 *         static constexpr const char *ServiceKey =
 *             "com.3dreconstruction.service.IReconstructionService";
 *         ...
 *     };
 *
 * The key is an explicit part of the interface contract (not derived from
 * RTTI), so lookups behave identically across compilers and across the
 * exe ↔ AppCore.dll ↔ plugin DLL boundaries — unlike typeid().name(),
 * whose mangling is implementation-defined and can differ per module.
 *
 * Two-tier surface — the registry a plugin receives is READ-ONLY:
 *   - `ServiceRegistry` exposes only get()/getRequired()/has(). It is what
 *     `IAppContext::services()` returns, so a plugin cannot unregister a
 *     service and install a replacement (that would defeat the "first
 *     registration wins" rule).
 *   - `MutableServiceRegistry` (src/core, composition root only) adds
 *     registerService()/unregister()/clear(). AppShell owns that type.
 *
 * Usage:
 *   // Register (AppShell — composition root only):
 *   m_registry->registerService<IReconstructionService>(m_reconService.get());
 *
 *   // Look up (Plugin::initialize):
 *   auto* svc = ctx->services()->get<IReconstructionService>();
 *
 * Lifetime: the registry does NOT own services; AppShell outlives all
 * consumers and clears the registry before destroying implementations.
 */
class ServiceRegistry {
public:
    ServiceRegistry() = default;
    ~ServiceRegistry() = default;

    // Non-copyable
    ServiceRegistry(const ServiceRegistry&) = delete;
    ServiceRegistry& operator=(const ServiceRegistry&) = delete;

    /**
     * @brief Tìm kiếm service theo interface type.
     * @return Pointer tới implementation, hoặc nullptr nếu chưa đăng ký.
     *
     * Prefer getRequired<T>() for services whose absence is a wiring bug
     * rather than a runtime condition.
     */
    template<typename Interface>
    Interface* get() const {
        static_assert(hasServiceKey<Interface>,
                      "Interface must declare 'static constexpr const char* ServiceKey'.");
        const QString key = keyOf<Interface>();
        std::shared_lock lock(m_mutex);
        auto it = m_services.find(key);
        if (it == m_services.end()) return nullptr;
        return static_cast<Interface*>(it.value());
    }

    /**
     * @brief Like get<T>(), but a missing service is treated as a fatal
     *        wiring bug (fail fast at the call site instead of deferring a
     *        nullptr crash into widget code).
     */
    template<typename Interface>
    Interface* getRequired() const {
        Interface* svc = get<Interface>();
        if (!svc) {
            qFatal("ServiceRegistry: required service %s is not registered. "
                   "It must be registered by the composition root (AppShell) "
                   "before consumers run.", keyOf<Interface>().toLocal8Bit().constData());
        }
        return svc;
    }

    /**
     * @brief Kiểm tra xem một service type có được đăng ký chưa.
     */
    template<typename Interface>
    bool has() const {
        static_assert(hasServiceKey<Interface>,
                      "Interface must declare 'static constexpr const char* ServiceKey'.");
        const QString key = keyOf<Interface>();
        std::shared_lock lock(m_mutex);
        return m_services.contains(key);
    }

protected:
    // ─── Write side: reachable only through MutableServiceRegistry (src/core) ─
    // AppShell owns the only MutableServiceRegistry in production. A plugin
    // receives the const ServiceRegistry* from IAppContext::services(), so the
    // capability below is not part of the plugin-facing contract.

    /**
     * @brief Đăng ký một service implementation theo interface type.
     * @tparam Interface  Interface có ServiceKey (e.g. IReconstructionService)
     * @param impl        Pointer tới implementation (không own, caller quản lý lifetime)
     *
     * Duplicate registration of the same interface is a wiring bug: it is
     * loudly reported and **refused**. The first registration wins, so a
     * later one (e.g. a plugin that registers a service the composition root
     * already installed) cannot silently take the service over.
     */
    template<typename Interface>
    void registerService(Interface* impl) {
        static_assert(hasServiceKey<Interface>,
                      "Interface must declare 'static constexpr const char* ServiceKey' "
                      "for ServiceRegistry registration.");
        static_assert(!std::is_const_v<Interface>,
                      "Cannot register a const interface pointer.");
        if (!impl) {
            qWarning() << "ServiceRegistry: refusing to register null service"
                       << keyOf<Interface>();
            return;
        }
        const QString key = keyOf<Interface>();
        std::unique_lock lock(m_mutex);
        if (m_services.contains(key)) {
            qWarning() << "ServiceRegistry: duplicate registration of"
                       << key << "- keeping the existing entry.";
            return;
        }
        m_services[key] = static_cast<void*>(impl);
    }

    /**
     * @brief Gỡ đăng ký một service (dùng khi cleanup).
     */
    template<typename Interface>
    void unregister() {
        static_assert(hasServiceKey<Interface>,
                      "Interface must declare 'static constexpr const char* ServiceKey'.");
        const QString key = keyOf<Interface>();
        std::unique_lock lock(m_mutex);
        m_services.remove(key);
    }

    /** @brief Xóa toàn bộ registrations. */
    void clear() {
        std::unique_lock lock(m_mutex);
        m_services.clear();
    }

private:
    template<typename T>
    static constexpr bool hasServiceKey =
        requires { { T::ServiceKey } -> std::convertible_to<const char*>; };

    template<typename Interface>
    static QString keyOf() {
        return QString::fromLatin1(Interface::ServiceKey);
    }

    mutable std::shared_mutex m_mutex;
    QHash<QString, void*> m_services;
};

#endif // SERVICEREGISTRY_H
