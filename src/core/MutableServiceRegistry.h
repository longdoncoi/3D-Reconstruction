#ifndef MUTABLESERVICEREGISTRY_H
#define MUTABLESERVICEREGISTRY_H

#include "ServiceRegistry.h"

/**
 * @brief Write side of the ServiceRegistry — composition root only.
 *
 * `ServiceRegistry` (src/include) is the plugin-facing, READ-ONLY view: it
 * only exposes get()/getRequired()/has(). This subclass re-exposes the
 * registration surface, and it lives in `src/core`, which is PRIVATE on
 * AppCore — so a plugin cannot name this type, let alone call it.
 *
 * Ownership stays with the composition root: AppShell holds the only
 * MutableServiceRegistry in production and the tests construct one when they
 * stand in for it. Consumers must never store a `MutableServiceRegistry*`;
 * they take the `const ServiceRegistry*` from `IAppContext::services()`.
 */
class MutableServiceRegistry : public ServiceRegistry {
public:
    using ServiceRegistry::registerService;
    using ServiceRegistry::unregister;
    using ServiceRegistry::clear;
};

#endif // MUTABLESERVICEREGISTRY_H
