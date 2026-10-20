#ifndef CRYPTO_UTILS_H
#define CRYPTO_UTILS_H

#include <QByteArray>
#include <QString>

#include "Global.h"

/**
 * @brief Stateless cryptographic primitives shared by the user/auth guard layer.
 *
 * Keeping the password format and the constant-time comparison out of
 * UserManager makes the security-sensitive part independently testable and
 * prevents a persistence refactor from silently changing the credential
 * contract.
 */
namespace CryptoUtils {

/// Hash a password in the versioned `pbkdf2$<iterations>$<salt>$<digest>` form.
APP_EXPORT QString hashPassword(const QString &password);

/// Verify a password against a PBKDF2 hash or a legacy bare SHA-256 hash.
APP_EXPORT bool verifyPassword(const QString &password, const QString &storedHash);

/// True when the stored value is the current PBKDF2 format (not legacy SHA-256).
APP_EXPORT bool isModernPasswordHash(const QString &storedHash);

/// Lowercase SHA-256 hex; used for legacy credentials and opaque INI buckets.
APP_EXPORT QString sha256Hex(const QString &value);

/// Compare two byte strings without an early exit on the first mismatch.
APP_EXPORT bool constantTimeEquals(const QByteArray &a, const QByteArray &b);

} // namespace CryptoUtils

#endif // CRYPTO_UTILS_H
