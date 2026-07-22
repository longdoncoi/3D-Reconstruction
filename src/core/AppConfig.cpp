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

    // Determine project root:
    // - Dev build: exe lives in build/<Config>/, CMakeLists.txt is two levels up.
    // - Production install: exe lives in the install folder itself; no CMakeLists.txt
    //   exists so we stay in that folder (all data dirs are siblings of the exe).
    QDir dir(m_appDir);
    int maxLevels = 4; // never climb more than 4 levels
    while (maxLevels-- > 0 && !dir.isRoot() && !dir.exists("CMakeLists.txt")) {
        dir.cdUp();
    }
    if (dir.exists("CMakeLists.txt")) {
        m_projectRoot = dir.absolutePath();
    } else {
        // Production install: treat the app directory itself as the root.
        m_projectRoot = m_appDir;
    }

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
