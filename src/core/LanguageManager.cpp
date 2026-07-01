#include "LanguageManager.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QCoreApplication>

// ─────────────────────────────────────────────────────────────────────────────
LanguageManager& LanguageManager::instance() {
    static LanguageManager inst;
    return inst;
}

LanguageManager::LanguageManager() {
    loadTranslations();
}

QString LanguageManager::currentLanguage() const {
    std::shared_lock lock(m_mutex);
    return m_lang;
}

void LanguageManager::setLanguage(const QString &lang) {
    if (lang != "vi" && lang != "en") {
        qWarning() << "[LanguageManager] Unknown language:" << lang;
        return;
    }
    
    QString newLang;
    {
        std::unique_lock lock(m_mutex);
        if (m_lang == lang) return;
        m_lang = lang;
        newLang = m_lang;
    }
    emit languageChanged(newLang);
}

QString LanguageManager::translate(const QString &key) const {
    std::shared_lock lock(m_mutex);
    if (!m_dict.contains(key)) {
        qWarning() << "[LanguageManager] Missing key:" << key;
        return key;
    }
    return m_lang == "vi" ? m_dict[key].vi : m_dict[key].en;
}

QString LanguageManager::translate(const char *key) const {
    if (!key) return QString();
    return this->translate(QString::fromUtf8(key));
}

QString LanguageManager::displayName() const {
    std::shared_lock lock(m_mutex);
    return m_lang == "vi" ? "Tiếng Việt" : "English";
}

QString LanguageManager::flagEmoji() const {
    std::shared_lock lock(m_mutex);
    // Unicode Regional Indicator flags
    return m_lang == "vi" ? "🇻🇳" : "🇬🇧";
}

// ─────────────────────────────────────────────────────────────────────────────
// Translation dictionary  (key → vi, en)
// ─────────────────────────────────────────────────────────────────────────────
void LanguageManager::loadTranslations() {
    std::unique_lock lock(m_mutex);
    m_dict.clear();
    QString appDir = QCoreApplication::applicationDirPath();
    
    // Load VI
    QFile fileVi(appDir + "/translations/translations_vi.json");
    if (fileVi.open(QIODevice::ReadOnly)) {
        QJsonObject objVi = QJsonDocument::fromJson(fileVi.readAll()).object();
        for (auto it = objVi.begin(); it != objVi.end(); ++it) {
            m_dict[it.key()].vi = it.value().toString();
        }
    } else {
        qWarning() << "[LanguageManager] Cannot open translation file:" << fileVi.fileName();
    }
    
    // Load EN
    QFile fileEn(appDir + "/translations/translations_en.json");
    if (fileEn.open(QIODevice::ReadOnly)) {
        QJsonObject objEn = QJsonDocument::fromJson(fileEn.readAll()).object();
        for (auto it = objEn.begin(); it != objEn.end(); ++it) {
            m_dict[it.key()].en = it.value().toString();
        }
    } else {
        qWarning() << "[LanguageManager] Cannot open translation file:" << fileEn.fileName();
    }
}
