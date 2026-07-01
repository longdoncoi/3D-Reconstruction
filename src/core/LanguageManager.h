#ifndef LANGUAGEMANAGER_H
#define LANGUAGEMANAGER_H

#include <QObject>
#include <QString>
#include <QMap>
#include <shared_mutex>
#include "Global.h"

/**
 * @brief Singleton quản lý ngôn ngữ ứng dụng (Tiếng Việt / English).
 *
 * Sử dụng:
 *   LM.tr("key")           — lấy chuỗi theo ngôn ngữ hiện tại
 *   LM.setLanguage("vi")   — chuyển sang tiếng Việt
 *   LM.setLanguage("en")   — chuyển sang tiếng Anh
 *
 * Ngôn ngữ được lưu per-user vào Config_User.ini:
 *   [User_<username>]
 *   language=vi
 */
class APP_EXPORT LanguageManager : public QObject {
    Q_OBJECT
public:
    static LanguageManager& instance();

    /** Trả về ngôn ngữ hiện tại: "vi" hoặc "en" */
    QString currentLanguage() const;

    /** Đổi ngôn ngữ và phát signal languageChanged() */
    void setLanguage(const QString &lang);

    /** Lấy chuỗi dịch theo key.
     *  Trả về key nếu không tìm thấy bản dịch (để dễ debug). */
    QString translate(const QString &key) const;
    QString translate(const char *key) const;

    /** Tên hiển thị của ngôn ngữ hiện tại */
    QString displayName() const;

    /** Unicode cờ của ngôn ngữ hiện tại */
    QString flagEmoji() const;

signals:
    void languageChanged(const QString &lang);

private:
    LanguageManager();
    ~LanguageManager() override = default;
    LanguageManager(const LanguageManager&) = delete;
    LanguageManager& operator=(const LanguageManager&) = delete;

    void loadTranslations();

    mutable std::shared_mutex m_mutex;
    QString m_lang = "vi"; // default: Vietnamese

    // Map: key → {vi, en}
    struct Entry { QString vi; QString en; };
    QMap<QString, Entry> m_dict;
};

// Convenience macros
#define LM LanguageManager::instance()
#define LM_TR(key) LanguageManager::instance().translate(key)

#endif // LANGUAGEMANAGER_H
