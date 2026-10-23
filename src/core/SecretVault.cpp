#include "SecretVault.h"

#include <QByteArray>
#include <QDebug>

#ifdef Q_OS_WIN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincrypt.h>
#endif

namespace {
constexpr const char *kEncodedPrefix = "enc1:";
constexpr const char *kPlainPrefix = "plain:";

/// Test seam: makes the protection call fail. Thread-local, so a test never
/// changes the behaviour of another thread.
thread_local bool g_protectionFails = false;

#ifdef Q_OS_WIN
/// Protects a value with DPAPI (CurrentUser scope). Returns an empty string
/// when the OS call fails — the caller decides what to do with that.
QString dpapiProtect(const QString &plainValue)
{
    const std::wstring wideValue = plainValue.toStdWString();
    DATA_BLOB input;
    input.pbData = reinterpret_cast<BYTE *>(const_cast<wchar_t *>(wideValue.data()));
    input.cbData = static_cast<DWORD>(wideValue.size() * sizeof(wchar_t));

    DATA_BLOB output;
    if (CryptProtectData(&input, L"3D-Reconstruction secret", nullptr, nullptr,
                         nullptr, CRYPTPROTECT_UI_FORBIDDEN, &output) && output.pbData) {
        const QByteArray blob(reinterpret_cast<const char *>(output.pbData),
                              static_cast<int>(output.cbData));
        LocalFree(output.pbData);
        return QString::fromLatin1(kEncodedPrefix) + blob.toBase64();
    }
    // The OS refused to protect the value.
    qWarning() << "SecretVault: CryptProtectData failed, error =" << GetLastError();
    return {};
}
#endif

} // namespace

void SecretVault::setProtectionFailureForTesting(bool fail)
{
    g_protectionFails = fail;
}

bool SecretVault::protectionFailsForTesting()
{
    return g_protectionFails;
}

QString SecretVault::encrypt(const QString &plainValue)
{
    if (plainValue.isEmpty()) return {};

#ifdef Q_OS_WIN
    if (g_protectionFails) {
        qWarning() << "SecretVault: the protection call was forced to fail (test seam).";
    } else {
        const QString sealed = dpapiProtect(plainValue);
        if (!sealed.isEmpty()) return sealed;
    }
#else
    qWarning() << "SecretVault: no OS protection available on this platform; "
                  "storing value marked as unencrypted plaintext.";
#endif
    // The value could not be protected: fail closed (see the class contract)
    // — never downgrade to cleartext just because DPAPI hiccuped.
    return protectionFailureFallback(plainValue);
}

QString SecretVault::protectionFailureFallback(const QString &plainValue)
{
    if (plainValue.isEmpty()) return {};
#ifdef Q_OS_WIN
    // Fail closed: this platform HAS an OS backend, so a protection failure
    // must not turn into a plaintext write. An empty result tells the caller
    // "do not write anything — keep the value that is already stored".
    return {};
#else
    // Documented platform limit: there is no OS backend at all on POSIX, so
    // the value is stored marked as explicitly unprotected.
    return QString::fromLatin1(kPlainPrefix) + plainValue;
#endif
}

SecretVault::Protection SecretVault::protectionOf(const QString &storedValue)
{
    if (storedValue.isEmpty()) return Protection::Empty;
    if (storedValue.startsWith(QString::fromLatin1(kEncodedPrefix))) return Protection::Protected;
    if (storedValue.startsWith(QString::fromLatin1(kPlainPrefix))) return Protection::PlaintextMarked;
    return Protection::LegacyPlaintext;
}

bool SecretVault::isUnprotected(const QString &storedValue)
{
    const Protection protection = protectionOf(storedValue);
    return protection == Protection::PlaintextMarked
        || protection == Protection::LegacyPlaintext;
}

QString SecretVault::decrypt(const QString &storedValue)
{
    if (storedValue.isEmpty()) return {};

    if (!storedValue.startsWith(QString::fromLatin1(kEncodedPrefix))) {
        if (storedValue.startsWith(QString::fromLatin1(kPlainPrefix)))
            return storedValue.mid(QString::fromLatin1(kPlainPrefix).size());
        // Legacy value written before the vault existed: return as-is so old
        // Config.ini files keep working until the next save migrates them.
        return storedValue;
    }

#ifdef Q_OS_WIN
    const QByteArray blob = QByteArray::fromBase64(
        storedValue.mid(QString::fromLatin1(kEncodedPrefix).size()).toLatin1());
    DATA_BLOB input;
    input.pbData = reinterpret_cast<BYTE *>(const_cast<char *>(blob.constData()));
    input.cbData = static_cast<DWORD>(blob.size());

    DATA_BLOB output;
    if (CryptUnprotectData(&input, nullptr, nullptr, nullptr, nullptr,
                           CRYPTPROTECT_UI_FORBIDDEN, &output) && output.pbData) {
        const wchar_t *wideData = reinterpret_cast<const wchar_t *>(output.pbData);
        const QString result = QString::fromWCharArray(
            wideData, static_cast<qsizetype>(output.cbData / sizeof(wchar_t)));
        LocalFree(output.pbData);
        return result;
    }
    // Blob unreadable (different user/machine): report empty rather than
    // surfacing garbage as if it were valid data.
    qWarning() << "SecretVault: CryptUnprotectData failed, error =" << GetLastError();
    return {};
#else
    return {};
#endif
}
