#ifndef OTP_STORE_H
#define OTP_STORE_H

#include <QDateTime>
#include <QMap>
#include <QString>

#include <mutex>

#include "Global.h"

/**
 * @brief Short-lived, single-use password-reset codes.
 *
 * The store owns its mutex so a caller cannot accidentally read/verify an OTP
 * while another thread is issuing one. Codes are compared in constant time
 * and removed on first use or expiry.
 */
class APP_EXPORT OtpStore {
public:
    /// Issue a six-digit CSPRNG code valid for @p ttlSeconds (default: 10 min).
    QString issue(const QString &email, int ttlSeconds = 600);

    /// Verify and consume a code. Wrong/expired/unknown codes return false.
    bool verify(const QString &email, const QString &otp);

    void clear();

private:
    struct Entry {
        QString otp;
        QDateTime expiry;
    };

    std::mutex m_mutex;
    QMap<QString, Entry> m_entries;
};

#endif // OTP_STORE_H
