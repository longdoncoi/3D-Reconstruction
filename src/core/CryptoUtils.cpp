#include "CryptoUtils.h"

#include <QCryptographicHash>
#include <QRandomGenerator>

namespace {

// Password storage format: "pbkdf2$<iterations>$<salt-hex>$<hash-hex>".
// Legacy values are bare SHA-256 hex strings and are transparently upgraded
// the next time the user proves the password at login.
constexpr const char *kPbkdf2Prefix = "pbkdf2$";
constexpr int kPbkdf2Iterations = 100000;
constexpr int kSaltBytes = 16;
constexpr char kHashPrefixShell = '$';

QByteArray hmacSha256(const QByteArray &key, const QByteArray &data)
{
    constexpr int blockSize = 64; // SHA-256 block size
    QByteArray k = key;
    if (k.size() > blockSize)
        k = QCryptographicHash::hash(k, QCryptographicHash::Sha256);
    k = k.leftJustified(blockSize, '\0');

    QByteArray inner(blockSize, '\0');
    QByteArray outer(blockSize, '\0');
    for (int i = 0; i < blockSize; ++i) {
        inner[i] = char(uint8_t(k[i]) ^ 0x36);
        outer[i] = char(uint8_t(k[i]) ^ 0x5c);
    }
    return QCryptographicHash::hash(
        outer + QCryptographicHash::hash(inner + data, QCryptographicHash::Sha256),
        QCryptographicHash::Sha256);
}

QByteArray pbkdf2HmacSha256(const QByteArray &password, const QByteArray &salt,
                            int iterations)
{
    constexpr int hashLen = 32;
    QByteArray derived;
    quint32 blockIndex = 1;
    while (derived.size() < hashLen) {
        QByteArray saltBlock = salt;
        saltBlock.append(char((blockIndex >> 24) & 0xff));
        saltBlock.append(char((blockIndex >> 16) & 0xff));
        saltBlock.append(char((blockIndex >> 8) & 0xff));
        saltBlock.append(char(blockIndex & 0xff));
        QByteArray u = hmacSha256(password, saltBlock);
        QByteArray t = u;
        for (int i = 1; i < iterations; ++i) {
            u = hmacSha256(password, u);
            for (int j = 0; j < u.size(); ++j)
                t[j] = char(uint8_t(t[j]) ^ uint8_t(u[j]));
        }
        derived += t;
        ++blockIndex;
    }
    return derived.left(hashLen);
}

} // namespace

namespace CryptoUtils {

QString hashPassword(const QString &password)
{
    quint32 words[kSaltBytes / 4];
    QRandomGenerator::system()->generate(words, words + kSaltBytes / 4);
    const QByteArray salt(reinterpret_cast<const char *>(words), sizeof(words));
    const QByteArray digest = pbkdf2HmacSha256(password.toUtf8(), salt, kPbkdf2Iterations);
    return QString::fromLatin1(kPbkdf2Prefix) + QString::number(kPbkdf2Iterations)
           + kHashPrefixShell + QString::fromLatin1(salt.toHex())
           + kHashPrefixShell + QString::fromLatin1(digest.toHex());
}

bool verifyPassword(const QString &password, const QString &storedHash)
{
    if (storedHash.startsWith(QString::fromLatin1(kPbkdf2Prefix))) {
        const QStringList parts =
            storedHash.mid(sizeof(kPbkdf2Prefix) - 1).split(kHashPrefixShell);
        if (parts.size() != 3) return false;
        bool okIterations = false;
        const int iterations = parts[0].toInt(&okIterations);
        const QByteArray salt = QByteArray::fromHex(parts[1].toLatin1());
        const QByteArray expected = QByteArray::fromHex(parts[2].toLatin1());
        if (!okIterations || iterations <= 0 || salt.size() < kSaltBytes / 2 || expected.isEmpty())
            return false;
        const QByteArray candidate = pbkdf2HmacSha256(password.toUtf8(), salt, iterations);
        return constantTimeEquals(expected, candidate);
    }
    // Legacy single-pass SHA-256 format; login() upgrades to PBKDF2
    // after this helper accepts the password.
    return storedHash == sha256Hex(password);
}

bool isModernPasswordHash(const QString &storedHash)
{
    return storedHash.startsWith(QString::fromLatin1(kPbkdf2Prefix));
}

QString sha256Hex(const QString &value)
{
    return QString::fromLatin1(QCryptographicHash::hash(
        value.toUtf8(), QCryptographicHash::Sha256).toHex());
}

bool constantTimeEquals(const QByteArray &a, const QByteArray &b)
{
    if (a.size() != b.size()) return false;
    uint8_t diff = 0;
    for (int i = 0; i < a.size(); ++i)
        diff |= uint8_t(a[i]) ^ uint8_t(b[i]);
    return diff == 0;
}

} // namespace CryptoUtils
