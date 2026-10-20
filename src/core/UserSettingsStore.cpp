#include "UserSettingsStore.h"

#include "SecretVault.h"

#include <QSettings>

QString UserSettingsStore::smtpSenderEmail() const
{
    return m_settings ? m_settings->value("SMTP/sender_email").toString() : QString();
}

QString UserSettingsStore::smtpSenderPassword() const
{
    if (!m_settings) return {};
    // Values written by older builds are legacy plaintext; the vault passes
    // them through unchanged so the next save() migrates them into "enc1:".
    return SecretVault::decrypt(m_settings->value("SMTP/sender_password").toString());
}

void UserSettingsStore::setSmtpCredentials(const QString &email, const QString &password)
{
    if (!m_settings) return;
    m_settings->setValue("SMTP/sender_email", email);
    m_settings->setValue("SMTP/sender_password", SecretVault::encrypt(password));
    m_settings->sync();
}

QString UserSettingsStore::getUserSecret(const QString &username, const QString &key,
                                         const QString &defaultVal) const
{
    if (!m_settings || username.isEmpty() || key.isEmpty()) return defaultVal;
    const QString value = m_settings->value("User_" + username + "/secret_" + key).toString();
    if (!value.isEmpty()) {
        const QString secret = SecretVault::decrypt(value);
        if (!secret.isEmpty()) return secret;
    }
    // Legacy plaintext preference written before secret storage existed;
    // migrated into "secret_<key>" the next time setUserSecret() runs.
    return m_settings->value("User_" + username + "/pref_" + key, defaultVal).toString();
}

void UserSettingsStore::setUserSecret(const QString &username, const QString &key,
                                      const QString &value)
{
    if (!m_settings || username.isEmpty() || key.isEmpty()) return;
    m_settings->setValue("User_" + username + "/secret_" + key, SecretVault::encrypt(value));
    // Retire any legacy plaintext copy of the same secret.
    m_settings->remove("User_" + username + "/pref_" + key);
    m_settings->sync();
}

QString UserSettingsStore::getUserPref(const QString &username, const QString &key,
                                       const QString &defaultVal) const
{
    if (!m_settings || username.isEmpty()) return defaultVal;
    return m_settings->value("User_" + username + "/pref_" + key, defaultVal).toString();
}

void UserSettingsStore::setUserPref(const QString &username, const QString &key,
                                    const QString &value)
{
    if (!m_settings || username.isEmpty()) return;
    m_settings->setValue("User_" + username + "/pref_" + key, value);
    m_settings->sync();
}
