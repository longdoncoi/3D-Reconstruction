#include "OtpStore.h"

#include "CryptoUtils.h"

#include <QRandomGenerator>

QString OtpStore::issue(const QString &email, int ttlSeconds)
{
    // Generate a 6-digit OTP with the system CSPRNG (global() is not
    // cryptographically secure and must not be used for reset codes).
    const int code = QRandomGenerator::system()->bounded(100000, 1000000);
    const QString otp = QString::number(code);

    std::lock_guard lock(m_mutex);
    m_entries[email.toLower()] = { otp, QDateTime::currentDateTime().addSecs(ttlSeconds) };
    return otp;
}

bool OtpStore::verify(const QString &email, const QString &otp)
{
    std::lock_guard lock(m_mutex);
    const QString key = email.toLower();
    auto it = m_entries.find(key);
    if (it == m_entries.end()) return false;
    if (QDateTime::currentDateTime() > it->expiry) {
        m_entries.erase(it);
        return false;
    }
    if (!CryptoUtils::constantTimeEquals(it->otp.toUtf8(), otp.toUtf8())) return false;
    m_entries.erase(it);
    return true;
}

void OtpStore::clear()
{
    std::lock_guard lock(m_mutex);
    m_entries.clear();
}
