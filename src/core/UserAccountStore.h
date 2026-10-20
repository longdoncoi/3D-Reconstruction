#ifndef USER_ACCOUNT_STORE_H
#define USER_ACCOUNT_STORE_H

#include "IUserManager.h"

#include <QMap>
#include <QString>
#include <QStringList>

#include "Global.h"

class QSettings;

/**
 * @brief Persistence boundary for accounts, profiles and password hashes.
 *
 * UserManager keeps the authentication/licence policy; this store owns the
 * INI shape, the in-memory account map and the credential map. It is not
 * thread-safe — every caller holds UserManager's database mutex.
 */
class APP_EXPORT UserAccountStore {
public:
    void attachSettings(QSettings *settings) { m_settings = settings; }

    void load();
    void save();
    void ensureDefaultAdmin(const QString &configPath);

    bool contains(const QString &username) const;
    UserInfo info(const QString &username) const;
    QStringList usernames() const;

    QString passwordHash(const QString &username) const;
    void setPasswordHash(const QString &username, const QString &hash);
    void insert(const UserInfo &info, const QString &hash);
    void update(const UserInfo &info);
    void remove(const QString &username);

    /// Username/email lookup used by login(); comparison is case-insensitive.
    QString findLoginUsername(const QString &usernameOrEmail) const;
    QString findUsernameByEmail(const QString &email) const;

private:
    QSettings *m_settings = nullptr;
    QMap<QString, UserInfo> m_users;
    QMap<QString, QString> m_passwordHashes; // username -> PBKDF2/legacy hash
};

#endif // USER_ACCOUNT_STORE_H
