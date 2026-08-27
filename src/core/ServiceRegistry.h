#ifndef SERVICEREGISTRY_H
#define SERVICEREGISTRY_H

#include <QHash>
#include <QString>
#include <typeinfo>
#include <shared_mutex>

/**
 * @brief Type-safe service registry (Service Locator pattern).
 *
 * Cho phép đăng ký và tìm kiếm service theo interface type mà không cần
 * biết concrete implementation.
 *
 * Cách dùng:
 *   // Đăng ký (trong AppShell::initializeServices):
 *   registry->registerService<IReconstructionService>(m_reconService.get());
 *
 *   // Tìm kiếm (trong Plugin::initialize):
 *   auto* svc = ctx->services()->get<IReconstructionService>();
 */
class ServiceRegistry {
public:
    ServiceRegistry() = default;
    ~ServiceRegistry() = default;

    // Non-copyable
    ServiceRegistry(const ServiceRegistry&) = delete;
    ServiceRegistry& operator=(const ServiceRegistry&) = delete;

    /**
     * @brief Đăng ký một service implementation theo interface type.
     * @tparam Interface  Interface class (e.g. IReconstructionService)
     * @param impl        Pointer tới implementation (không own, caller quản lý lifetime)
     */
    template<typename Interface>
    void registerService(Interface* impl) {
        std::unique_lock lock(m_mutex);
        m_services[QString(typeid(Interface).name())] =
            static_cast<void*>(impl);
    }

    /**
     * @brief Tìm kiếm service theo interface type.
     * @tparam Interface  Interface class cần tìm
     * @return Pointer tới implementation, hoặc nullptr nếu chưa đăng ký.
     */
    template<typename Interface>
    Interface* get() const {
        std::shared_lock lock(m_mutex);
        auto it = m_services.find(QString(typeid(Interface).name()));
        if (it == m_services.end()) return nullptr;
        return static_cast<Interface*>(it.value());
    }

    /**
     * @brief Kiểm tra xem một service type có được đăng ký chưa.
     */
    template<typename Interface>
    bool has() const {
        std::shared_lock lock(m_mutex);
        return m_services.contains(QString(typeid(Interface).name()));
    }

    /**
     * @brief Gỡ đăng ký một service (dùng khi cleanup).
     */
    template<typename Interface>
    void unregister() {
        std::unique_lock lock(m_mutex);
        m_services.remove(QString(typeid(Interface).name()));
    }

    /** @brief Xóa toàn bộ registrations. */
    void clear() {
        std::unique_lock lock(m_mutex);
        m_services.clear();
    }

private:
    mutable std::shared_mutex m_mutex;
    QHash<QString, void*> m_services;
};

#endif // SERVICEREGISTRY_H
