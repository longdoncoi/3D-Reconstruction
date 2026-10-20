#ifndef USER_SETTINGS_STORE_H
#define USER_SETTINGS_STORE_H

#include <QString>

#include "Global.h"

class QSettings;

/**
 * @brief SMTP credentials and per-user preferences/secrets in the user INI.
 *
 * Not thread-safe by design: UserManager serialises every call with its
 * database mutex, so a preference update cannot race a user/licence write.
 */
class APP_EXPORT UserSettingsStore {
public:
    void attachSettings(QSettings *settings) { m_settings = settings; }

    QString smtpSenderEmail() const;
    QString smtpSenderPassword() const;
    void setSmtpCredentials(const QString &email, const QString &password);

    QString getUserSecret(const QString &username, const QString &key,
                          const QString &defaultVal = {}) const;
    void setUserSecret(const QString &username, const QString &key, const QString &value);

    QString getUserPref(const QString &username, const QString &key,
                        const QString &defaultVal = {}) const;
    void setUserPref(const QString &username, const QString &key, const QString &value);

private:
    QSettings *m_settings = nullptr;
};

#endif // USER_SETTINGS_STORE_H
