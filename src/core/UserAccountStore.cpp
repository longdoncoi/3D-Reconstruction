#include "UserAccountStore.h"

#include "CryptoUtils.h"

#include <QDate>
#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QRandomGenerator>
#include <QSettings>

void UserAccountStore::load()
{
    m_users.clear();
    m_passwordHashes.clear();
    if (!m_settings) return;

    const QStringList names = m_settings->value("Users/usernames").toStringList();
    for (const QString &name : names) {
        const QString section = "User_" + name;
        UserInfo user;
        user.username = name;
        m_passwordHashes.insert(name, m_settings->value(section + "/password_hash").toString());
        user.email = m_settings->value(section + "/email").toString();
        user.role = m_settings->value(section + "/role", "User").toString() == "Admin"
                        ? UserRole::Admin : UserRole::User;
        user.licenseKey = m_settings->value(section + "/license_key").toString();
        user.licenseStatus = m_settings->value(section + "/license_status", "None").toString();
        const QString expiry = m_settings->value(section + "/license_expiry").toString();
        const QString trial = m_settings->value(section + "/trial_start").toString();
        if (!expiry.isEmpty()) user.licenseExpiry = QDate::fromString(expiry, Qt::ISODate);
        if (!trial.isEmpty()) user.trialStart = QDate::fromString(trial, Qt::ISODate);
        user.avatarPath = m_settings->value(section + "/avatar_path").toString();
        m_users.insert(name, user);
    }
}

void UserAccountStore::save()
{
    if (!m_settings) return;

    const QStringList names = m_users.keys();
    m_settings->setValue("Users/count", names.size());
    m_settings->setValue("Users/usernames", names);
    for (const QString &name : names) {
        const UserInfo &user = m_users[name];
        const QString section = "User_" + name;
        m_settings->setValue(section + "/password_hash",
                             m_passwordHashes.value(name));
        m_settings->setValue(section + "/email", user.email);
        m_settings->setValue(section + "/role",
                             user.role == UserRole::Admin ? "Admin" : "User");
        m_settings->setValue(section + "/license_key", user.licenseKey);
        m_settings->setValue(section + "/license_status", user.licenseStatus);
        m_settings->setValue(section + "/license_expiry",
                             user.licenseExpiry.isValid()
                                 ? user.licenseExpiry.toString(Qt::ISODate) : "");
        m_settings->setValue(section + "/trial_start",
                             user.trialStart.isValid()
                                 ? user.trialStart.toString(Qt::ISODate) : "");
        m_settings->setValue(section + "/avatar_path", user.avatarPath);
    }
}

void UserAccountStore::ensureDefaultAdmin(const QString &configPath)
{
    if (m_users.contains("admin"))
        return;

    // The initial credential comes from the installer/environment. If it was
    // not provisioned, generate a high-entropy password and write it to an
    // owner-only file next to the user database instead of hard-coding a
    // well-known password (the old behaviour was "1") or leaking it into logs.
    QString initialPassword = qEnvironmentVariable("APP_ADMIN_INITIAL_PASSWORD").trimmed();
    const bool generated = initialPassword.isEmpty();
    if (generated) {
        constexpr char alphabet[] =
            "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%^&*";
        constexpr int alphabetSize = static_cast<int>(sizeof(alphabet)) - 1;
        constexpr int passwordLength = 20;
        initialPassword.reserve(passwordLength);
        for (int i = 0; i < passwordLength; ++i)
            initialPassword.append(
                QLatin1Char(alphabet[QRandomGenerator::system()->bounded(alphabetSize)]));
    }

    UserInfo admin;
    admin.username = "admin";
    admin.email = "admin@example.com";
    admin.role = UserRole::Admin;
    admin.licenseStatus = "Activated";
    admin.licenseExpiry = QDate::currentDate().addYears(10);
    insert(admin, CryptoUtils::hashPassword(initialPassword));

    if (!generated) {
        qDebug() << "[UserAuth] Default admin account created from APP_ADMIN_INITIAL_PASSWORD.";
        return;
    }

    const QString passwordFile =
        QFileInfo(configPath).absolutePath() + "/admin_initial_password.txt";
    QFile file(passwordFile);
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        file.write(QStringLiteral(
            "Initial admin password: %1\n"
            "Change it after the first login, then delete this file.\n")
                       .arg(initialPassword)
                       .toUtf8());
        file.close();
        QFile::setPermissions(passwordFile,
                              QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        qWarning().noquote()
            << "[UserAuth] Generated initial admin password; see" << passwordFile;
    } else {
        // No safe channel for the generated credential. Refuse to fall back to
        // a known password; the operator must provision one explicitly.
        setPasswordHash("admin", CryptoUtils::hashPassword(
            QString::fromLatin1("unprovisioned-") +
            QString::number(QRandomGenerator::system()->generate())));
        qWarning().noquote()
            << "[UserAuth] Could not write the initial admin password file."
            << "Set APP_ADMIN_INITIAL_PASSWORD before first run.";
    }
}

bool UserAccountStore::contains(const QString &username) const
{
    return m_users.contains(username);
}

UserInfo UserAccountStore::info(const QString &username) const
{
    return m_users.value(username);
}

QStringList UserAccountStore::usernames() const
{
    return m_users.keys();
}

QString UserAccountStore::passwordHash(const QString &username) const
{
    return m_passwordHashes.value(username);
}

void UserAccountStore::setPasswordHash(const QString &username, const QString &hash)
{
    m_passwordHashes[username] = hash;
}

void UserAccountStore::insert(const UserInfo &info, const QString &hash)
{
    UserInfo user = info;
    m_passwordHashes[user.username] = hash;
    m_users[user.username] = user;
}

void UserAccountStore::update(const UserInfo &info)
{
    m_users[info.username] = info;
}

void UserAccountStore::remove(const QString &username)
{
    m_users.remove(username);
    m_passwordHashes.remove(username);
    if (m_settings) m_settings->remove("User_" + username);
}

QString UserAccountStore::findLoginUsername(const QString &usernameOrEmail) const
{
    const QString lowerInput = usernameOrEmail.toLower();
    for (auto it = m_users.begin(); it != m_users.end(); ++it) {
        if (it.value().username.toLower() == lowerInput ||
            it.value().email.toLower() == lowerInput) {
            return it.key();
        }
    }
    return {};
}

QString UserAccountStore::findUsernameByEmail(const QString &email) const
{
    const QString lowerEmail = email.toLower();
    for (auto it = m_users.begin(); it != m_users.end(); ++it) {
        if (it.value().email.toLower() == lowerEmail)
            return it.key();
    }
    return {};
}
