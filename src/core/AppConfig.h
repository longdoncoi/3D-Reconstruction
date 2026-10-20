#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <QString>
#include <QDir>
#include <QFileInfo>
#include <shared_mutex>
#include "Global.h" // For APP_EXPORT
#include "IAppPaths.h"

/**
 * @brief Process-wide path provider. Implements IAppPaths so plugins and
 * services can resolve it from the ServiceRegistry; the bootstrap accessor
 * serves code (Main.cpp / AppShell / Logger) that runs before the registry
 * exists.
 */
class APP_EXPORT AppConfig : public IAppPaths {
public:
    /**
     * @brief Bootstrap / composition-root accessor.
     *
     * Allowed only in Main.cpp, AppShell.cpp and Logger.cpp — the code that
     * runs before (or while wiring) the ServiceRegistry. Everywhere else,
     * resolve IAppPaths from the registry.
     */
    static AppConfig& bootstrap();

public:
    void initialize(const QString& appDir);

public:
    QString appDir() const override;
    QString projectRootDir() const override;
    QString dataRootDir() const override;
    QString configPath() const override;
    QString logsDir() const override;
    QString modelsDir() const override;
    QString predictDir(const QString& type) const override;
    QString aiAssistantDir() const override;
    QString aiComputerVisionDir() const override;
    QString uploadDir() const override;
    QString thumbnailsDir() const override;
    QString pluginsDir() const override;

    // ── Runtime configuration (composition-root only) ─────────────────────────
    // Not part of IAppPaths: plugins/services receive these values injected by
    // AppShell, so this surface stays a bootstrap concern.
    /// Python interpreter used to launch the AI assistant backend.
    /// Config: [AI] python_interpreter=<path-or-command> (default "python").
    QString pythonInterpreter() const;
    /// Port the AI assistant backend listens on.
    /// Config: [AI] server_port=<1024-65535> (default 8080).
    int aiServerPort() const;

private:
    AppConfig() = default;
    ~AppConfig() = default;
    
    // Non-copyable
    AppConfig(const AppConfig&) = delete;
    AppConfig& operator=(const AppConfig&) = delete;

    mutable std::shared_mutex m_mutex;
    QString m_appDir;
    QString m_projectRoot; // Cache project root for convenience if running from build dir
    QString m_dataRoot;    // Writeable root directory (AppData in production)
};

#endif // APPCONFIG_H
