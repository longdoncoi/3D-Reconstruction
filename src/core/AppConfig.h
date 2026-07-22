#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <QString>
#include <QDir>
#include <QFileInfo>
#include <shared_mutex>
#include "Global.h" // For APP_EXPORT

class APP_EXPORT AppConfig {
public:
    static AppConfig& instance();

    void initialize(const QString& appDir);

    QString appDir() const;
    QString configPath() const;
    QString logsDir() const;
    QString modelsDir() const;
    QString predictDir(const QString& type) const;
    QString aiTrainingDir() const;
    QString uploadDir() const;
    QString pluginsDir() const;

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
