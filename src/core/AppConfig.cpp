#include "AppConfig.h"
#include <QDir>
#include <QApplication>
#include <QDebug>

AppConfig& AppConfig::instance() {
    static AppConfig instance;
    return instance;
}

void AppConfig::initialize(const QString& appDir) {
    std::unique_lock lock(m_mutex);
    m_appDir = appDir;
    
    // Find project root by looking for CMakeLists.txt
    QDir dir(m_appDir);
    while (!dir.isRoot() && !dir.exists("CMakeLists.txt")) {
        dir.cdUp();
    }
    m_projectRoot = dir.absolutePath();

    QFileInfo configInfo(QDir::cleanPath(m_projectRoot + "/Config/Config.ini"));
    QDir configDir = configInfo.absoluteDir();
    if (!configDir.exists() && !configDir.mkpath(".")) {
        qCritical() << "AppConfig: Failed to create config directory:" << configDir.absolutePath();
    }
}

QString AppConfig::appDir() const {
    std::shared_lock lock(m_mutex);
    return m_appDir;
}

QString AppConfig::configPath() const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_projectRoot + "/Config/Config.ini");
}

QString AppConfig::logsDir() const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_projectRoot + "/Logs");
}

QString AppConfig::modelsDir() const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_projectRoot + "/AITraining/Models");
}

QString AppConfig::predictDir(const QString& type) const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_projectRoot + "/Predict/" + type);
}

QString AppConfig::aiTrainingDir() const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_projectRoot + "/AITraining");
}

QString AppConfig::uploadDir() const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_projectRoot + "/Upload");
}

QString AppConfig::pluginsDir() const {
    std::shared_lock lock(m_mutex);
    return QDir::cleanPath(m_appDir + "/plugins");
}
