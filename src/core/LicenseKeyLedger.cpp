#include "LicenseKeyLedger.h"

#include "CryptoUtils.h"
#include "SecretVault.h"

#include <QSettings>
#include <QRandomGenerator>

void LicenseKeyLedger::clear()
{
    m_keys.clear();
    m_owners.clear();
}

void LicenseKeyLedger::load()
{
    clear();
    if (!m_settings) return;

    const QStringList storedKeys = m_settings->value("LicenseKeys/keys").toStringList();
    for (const QString &stored : storedKeys) {
        // Values may be DPAPI-encrypted (enc1:...) or legacy plaintext; the
        // vault passes the latter through so old configs migrate transparently.
        const QString key = SecretVault::decrypt(stored);
        if (key.isEmpty()) continue;
        // Subkey path uses a hash so the plaintext key never appears in the INI.
        const QString bucket = CryptoUtils::sha256Hex(key);
        bool used = m_settings->value("LicenseKeys/" + bucket + "_used", false).toBool();
        QString keyOwner = m_settings->value("LicenseKeys/" + bucket + "_user").toString();
        // Legacy format used the plaintext key directly in the path — migrate it.
        if (!m_settings->contains("LicenseKeys/" + bucket + "_used") &&
            m_settings->contains("LicenseKeys/" + key + "_used")) {
            used = m_settings->value("LicenseKeys/" + key + "_used", false).toBool();
            keyOwner = m_settings->value("LicenseKeys/" + key + "_user").toString();
            m_settings->remove("LicenseKeys/" + key + "_used");
            m_settings->remove("LicenseKeys/" + key + "_user");
        }
        m_keys[key] = used;
        if (!keyOwner.isEmpty()) m_owners[key] = keyOwner;
    }
}

void LicenseKeyLedger::save()
{
    if (!m_settings) return;

    QStringList storedKeys;
    const QStringList keys = m_keys.keys();
    for (const QString &key : keys) {
        // Store DPAPI-encrypted; the path bucket is a hash of the key.
        storedKeys.append(SecretVault::encrypt(key));
        const QString bucket = CryptoUtils::sha256Hex(key);
        m_settings->setValue("LicenseKeys/" + bucket + "_used", m_keys[key]);
        m_settings->setValue("LicenseKeys/" + bucket + "_user", m_owners.value(key));
    }
    m_settings->setValue("LicenseKeys/keys", storedKeys);
}

QString LicenseKeyLedger::generate()
{
    const QString chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // exclude ambiguous 0,1,I,O
    QString key;
    for (int group = 0; group < 4; ++group) {
        for (int i = 0; i < 5; ++i)
            key += chars[QRandomGenerator::system()->bounded(chars.size())];
        if (group < 3) key += '-';
    }
    m_keys[key] = false;
    return key;
}

QStringList LicenseKeyLedger::available() const
{
    return m_keys.keys();
}

bool LicenseKeyLedger::contains(const QString &key) const
{
    return m_keys.contains(key);
}

bool LicenseKeyLedger::isUsed(const QString &key) const
{
    return m_keys.value(key, false);
}

QString LicenseKeyLedger::owner(const QString &key) const
{
    return m_owners.value(key);
}

bool LicenseKeyLedger::claim(const QString &key, const QString &username)
{
    if (!m_keys.contains(key) || m_keys.value(key)) return false;
    m_keys[key] = true;
    m_owners[key] = username;
    return true;
}

bool LicenseKeyLedger::release(const QString &key, QString &previousOwner)
{
    if (!m_keys.contains(key)) return false;

    previousOwner = m_owners.take(key);
    m_keys.remove(key);

    if (m_settings) {
        const QString bucket = CryptoUtils::sha256Hex(key);
        m_settings->remove("LicenseKeys/" + bucket + "_used");
        m_settings->remove("LicenseKeys/" + bucket + "_user");
        // Legacy plaintext-path entries (pre-encryption configs)
        m_settings->remove("LicenseKeys/" + key + "_used");
        m_settings->remove("LicenseKeys/" + key + "_user");
    }
    return true;
}
