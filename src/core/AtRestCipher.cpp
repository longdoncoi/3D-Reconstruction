#include "AtRestCipher.h"

#include <QByteArray>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRandomGenerator>
#include <QString>

#ifdef Q_OS_WIN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

/// The key file is read through a helper that is never inlined: the AppCore
/// unity build would fold it into loadKey(), and the coverage exporter would
/// then report the helper's own lines as never executed (see
/// tests/README.md, "Unity build vs. measurability").
#if defined(_MSC_VER)
#  define ATREST_NOINLINE __declspec(noinline)
#else
#  define ATREST_NOINLINE
#endif

/// Envelope marker — mirrors the Python SEALED_BINARY_PREFIX.
const char kMarker[] = "enc:v1:";
/// Associated-data prefix — mirrors the Python _AAD_PREFIX.
const char kAadPrefix[] = "ai_assistant.local_state.v1:";
/// AES-256 key length in bytes.
constexpr int kKeyBytes = 32;
/// GCM nonce length in bytes (96-bit, the GCM-native size).
constexpr int kNonceBytes = 12;
/// GCM authentication tag length in bytes.
constexpr int kTagBytes = 16;

/// Read an environment variable the way the Python settings module does:
/// trimmed, empty means "not configured".
QString envValue(const char *name)
{
    return qEnvironmentVariable(name).trimmed();
}

/// Expand a leading "~" the way pathlib.Path.expanduser() does.
QString expandUser(const QString &path)
{
    if (path == QLatin1String("~")) return QDir::homePath();
    if (path.startsWith(QLatin1String("~/")) || path.startsWith(QLatin1String("~\\")))
        return QDir::homePath() + path.mid(1);
    return path;
}

/// Decode a base64 payload (standard or URL-safe alphabet).
///
/// Mirrors base64.b64decode(text, altchars=b"-_", validate=True): the
/// alphabet is validated, the padding is completed to a multiple of four and
/// a padding run longer than two characters is rejected.
bool decodeBase64(const QByteArray &text, QByteArray *out)
{
    QByteArray normalized;
    normalized.reserve(static_cast<qsizetype>(text.size()));
    int padding = 0;
    for (const char c : text) {
        if (c == '=') {
            ++padding;
            continue;
        }
        if (padding > 0) return false; // data after the padding
        char mapped = c;
        if (mapped == '-') {
            mapped = '+';
        } else if (mapped == '_') {
            mapped = '/';
        } else if (!(c == '+' || c == '/' || (c >= 'A' && c <= 'Z') ||
                     (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
            return false;
        }
        normalized.append(mapped);
    }
    // The Python decoder completes the padding to a multiple of four and then
    // rejects more than two '=' characters.
    const qsizetype extra = (4 - ((normalized.size() + padding) % 4)) % 4;
    if (padding + extra > 2) return false;
    normalized.append(QByteArray(padding + extra, '='));
    const QByteArray decoded = QByteArray::fromBase64(
        normalized, QByteArray::Base64Encoding | QByteArray::AbortOnBase64DecodingErrors);
    if (decoded.isEmpty() && !normalized.isEmpty()) return false;
    *out = decoded;
    return true;
}

/// Decode a hexadecimal payload. Mirrors bytes.fromhex().
bool decodeHex(const QByteArray &text, QByteArray *out)
{
    QByteArray digits;
    digits.reserve(static_cast<qsizetype>(text.size()));
    for (const char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
            continue;
        const bool isDigit = c >= '0' && c <= '9';
        const bool isLower = c >= 'a' && c <= 'f';
        const bool isUpper = c >= 'A' && c <= 'F';
        if (!isDigit && !isLower && !isUpper) return false;
        digits.append(c);
    }
    if (digits.size() % 2 != 0) return false;
    *out = QByteArray::fromHex(digits);
    return true;
}

/// Parse a key literal. Mirrors the Python _decode_key(): "hex:<hex>",
/// "base64:<base64>" or a bare base64 value, exactly 32 bytes.
bool parseKeyLiteral(const QByteArray &raw, QByteArray *out)
{
    const QByteArray text = raw.trimmed();
    QByteArray key;
    if (text.startsWith("hex:")) {
        if (!decodeHex(text.mid(4), &key)) return false;
    } else {
        QByteArray encoded = text;
        if (encoded.startsWith("base64:")) encoded = encoded.mid(7);
        if (!decodeBase64(encoded, &key)) return false;
    }
    if (key.size() != kKeyBytes) return false;
    *out = key;
    return true;
}

/// Where the generated key lives when no explicit key is configured.
QString defaultKeyFilePath()
{
#ifdef Q_OS_WIN
    QString base = envValue("LOCALAPPDATA");
    if (base.isEmpty()) base = envValue("APPDATA");
    if (base.isEmpty()) base = QDir::homePath() + QStringLiteral("/AppData/Local");
#else
    QString base = envValue("XDG_CONFIG_HOME");
    if (base.isEmpty()) base = QDir::homePath() + QStringLiteral("/.config");
#endif
    return QDir(expandUser(base)).filePath(QStringLiteral("3d-reconstruction/ai_assistant.key"));
}

/// The explicitly configured key file (empty when none is configured).
QString explicitKeyFilePath()
{
    const QString raw = envValue("AI_STORAGE_KEY_FILE");
    if (raw.isEmpty()) return {};
    return expandUser(raw);
}

/// Random bytes from the OS CSPRNG.
QByteArray randomBytes(int count)
{
    QByteArray bytes(count, '\0');
#ifdef Q_OS_WIN
    const NTSTATUS status = BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(bytes.data()),
                                             static_cast<ULONG>(count),
                                             BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (!BCRYPT_SUCCESS(status)) return {};
#else
    for (int i = 0; i < count; ++i)
        bytes[i] = static_cast<char>(QRandomGenerator::system()->bounded(256));
#endif
    return bytes;
}

/// Create the key file with an exclusive create: an existing file is never
/// overwritten, the caller reads it instead.
bool createKeyFile(const QString &path, const QByteArray &payload)
{
    const QString parent = QFileInfo(path).absolutePath();
    if (!QDir().mkpath(parent)) return false;
#ifdef Q_OS_UNIX
    const QByteArray native = QFile::encodeName(path);
    const int descriptor = ::open(native.constData(), O_WRONLY | O_CREAT | O_EXCL,
                                  S_IRUSR | S_IWUSR);
    if (descriptor < 0) return false;
    const ssize_t written = ::write(descriptor, payload.constData(),
                                    static_cast<size_t>(payload.size()));
    ::close(descriptor);
    return written == static_cast<ssize_t>(payload.size());
#else
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::NewOnly)) return false;
    const bool complete = file.write(payload) == payload.size();
    file.close();
    return complete;
#endif
}

/// Read the key file and decode it.
ATREST_NOINLINE bool readKeyFile(const QString &path, QByteArray *out)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return false;
    const QByteArray text = file.readAll();
    file.close();
    return parseKeyLiteral(text, out);
}

/// Resolve the at-rest key. Mirrors the Python load_storage_cipher(): it is
/// fail-closed, a configured key is never replaced by a generated one.
bool loadKey(QByteArray *out)
{
    const QString configured = envValue("AI_STORAGE_KEY");
    if (!configured.isEmpty()) return parseKeyLiteral(configured.toUtf8(), out);

    const QString explicitPath = explicitKeyFilePath();
    const bool isExplicit = !explicitPath.isEmpty();
    const QString path = isExplicit ? explicitPath : defaultKeyFilePath();

    if (QFileInfo::exists(path)) return readKeyFile(path, out);
    if (isExplicit) return false; // a configured key file must exist

    const QByteArray key = randomBytes(kKeyBytes);
    if (key.size() != kKeyBytes) return false;
    // "base64:<url-safe base64>\n" — the exact layout the Python service
    // writes, so both sides read the same key file.
    const QByteArray payload = QByteArray("base64:")
                               + key.toBase64(QByteArray::Base64UrlEncoding)
                               + QByteArray("\n");
    if (createKeyFile(path, payload)) { *out = key; return true; }
    // Another process created the file first: its key is the one on disk.
    return readKeyFile(path, out);
}

/// The associated data binds the context into the AEAD.
QByteArray associatedData(const QString &context)
{
    return QByteArray(kAadPrefix) + context.toUtf8();
}

#ifdef Q_OS_WIN

/// A BCrypt AES-256-GCM key handle.
struct GcmKey {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_KEY_HANDLE key = nullptr;
    QByteArray keyObject;
};

/// Release the BCrypt handles.
void closeGcmKey(GcmKey *gcm)
{
    if (gcm->key) {
        BCryptDestroyKey(gcm->key);
        gcm->key = nullptr;
    }
    if (gcm->algorithm) {
        BCryptCloseAlgorithmProvider(gcm->algorithm, 0);
        gcm->algorithm = nullptr;
    }
}

/// Open an AES-256-GCM key handle for the given key material.
bool openGcmKey(const QByteArray &keyMaterial, GcmKey *gcm)
{
    NTSTATUS status = BCryptOpenAlgorithmProvider(&gcm->algorithm, BCRYPT_AES_ALGORITHM,
                                                   nullptr, 0);
    if (!BCRYPT_SUCCESS(status)) { closeGcmKey(gcm); return false; }

    const wchar_t *mode = BCRYPT_CHAIN_MODE_GCM;
    status = BCryptSetProperty(gcm->algorithm, BCRYPT_CHAINING_MODE,
                               reinterpret_cast<PUCHAR>(const_cast<wchar_t *>(mode)),
                               static_cast<ULONG>(sizeof(BCRYPT_CHAIN_MODE_GCM)), 0);
    if (!BCRYPT_SUCCESS(status)) { closeGcmKey(gcm); return false; }

    DWORD objectLength = 0;
    DWORD resultLength = 0;
    status = BCryptGetProperty(gcm->algorithm, BCRYPT_OBJECT_LENGTH,
                               reinterpret_cast<PUCHAR>(&objectLength),
                               static_cast<ULONG>(sizeof(objectLength)), &resultLength, 0);
    if (!BCRYPT_SUCCESS(status)) { closeGcmKey(gcm); return false; }

    gcm->keyObject = QByteArray(static_cast<int>(objectLength), '\0');
    status = BCryptGenerateSymmetricKey(
        gcm->algorithm, &gcm->key, reinterpret_cast<PUCHAR>(gcm->keyObject.data()),
        objectLength,
        reinterpret_cast<PUCHAR>(const_cast<char *>(keyMaterial.constData())),
        static_cast<ULONG>(keyMaterial.size()), 0);
    if (!BCRYPT_SUCCESS(status)) { closeGcmKey(gcm); return false; }
    return true;
}

/// Encrypt with AES-256-GCM; the authentication tag is written into `tag`.
bool gcmEncrypt(const QByteArray &keyMaterial, const QByteArray &nonce, const QByteArray &aad,
                const QByteArray &plain, QByteArray *cipher, QByteArray *tag)
{
    GcmKey gcm;
    if (!openGcmKey(keyMaterial, &gcm)) { closeGcmKey(&gcm); return false; }

    char tagBuffer[kTagBytes] = {};
    BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info;
    ZeroMemory(&info, sizeof(info));
    info.cbSize = static_cast<DWORD>(sizeof(info));
    info.dwInfoVersion = BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO_VERSION;
    info.pbNonce = reinterpret_cast<PUCHAR>(const_cast<char *>(nonce.constData()));
    info.cbNonce = static_cast<ULONG>(nonce.size());
    info.pbAuthData = reinterpret_cast<PUCHAR>(const_cast<char *>(aad.constData()));
    info.cbAuthData = static_cast<ULONG>(aad.size());
    info.pbTag = reinterpret_cast<PUCHAR>(tagBuffer);
    info.cbTag = kTagBytes;

    // GCM is a stream mode: the ciphertext has the length of the plaintext.
    QByteArray buffer(qMax<qsizetype>(1, plain.size()), '\0');
    ULONG written = 0;
    const NTSTATUS status = BCryptEncrypt(
        gcm.key,
        reinterpret_cast<PUCHAR>(const_cast<char *>(plain.constData())),
        static_cast<ULONG>(plain.size()), reinterpret_cast<PUCHAR>(&info),
        reinterpret_cast<PUCHAR>(const_cast<char *>(nonce.constData())),
        static_cast<ULONG>(nonce.size()), reinterpret_cast<PUCHAR>(buffer.data()),
        static_cast<ULONG>(buffer.size()), &written, 0);
    closeGcmKey(&gcm);
    if (!BCRYPT_SUCCESS(status)) return false;

    *cipher = buffer.left(static_cast<int>(written));
    *tag = QByteArray(tagBuffer, kTagBytes);
    return true;
}

/// Decrypt with AES-256-GCM; a wrong tag fails the authentication.
bool gcmDecrypt(const QByteArray &keyMaterial, const QByteArray &nonce, const QByteArray &aad,
                const QByteArray &cipher, const QByteArray &tag, QByteArray *plain)
{
    GcmKey gcm;
    if (!openGcmKey(keyMaterial, &gcm)) { closeGcmKey(&gcm); return false; }

    BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info;
    ZeroMemory(&info, sizeof(info));
    info.cbSize = static_cast<DWORD>(sizeof(info));
    info.dwInfoVersion = BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO_VERSION;
    info.pbNonce = reinterpret_cast<PUCHAR>(const_cast<char *>(nonce.constData()));
    info.cbNonce = static_cast<ULONG>(nonce.size());
    info.pbAuthData = reinterpret_cast<PUCHAR>(const_cast<char *>(aad.constData()));
    info.cbAuthData = static_cast<ULONG>(aad.size());
    info.pbTag = reinterpret_cast<PUCHAR>(const_cast<char *>(tag.constData()));
    info.cbTag = static_cast<ULONG>(tag.size());

    QByteArray buffer(qMax<qsizetype>(1, cipher.size()), '\0');
    ULONG written = 0;
    const NTSTATUS status = BCryptDecrypt(
        gcm.key,
        reinterpret_cast<PUCHAR>(const_cast<char *>(cipher.constData())),
        static_cast<ULONG>(cipher.size()), reinterpret_cast<PUCHAR>(&info),
        reinterpret_cast<PUCHAR>(const_cast<char *>(nonce.constData())),
        static_cast<ULONG>(nonce.size()), reinterpret_cast<PUCHAR>(buffer.data()),
        static_cast<ULONG>(buffer.size()), &written, 0);
    closeGcmKey(&gcm);
    if (!BCRYPT_SUCCESS(status)) return false;

    *plain = buffer.left(static_cast<int>(written));
    return true;
}

#endif // Q_OS_WIN

} // namespace

QByteArray AtRestCipher::marker()
{
    return QByteArray(kMarker);
}

bool AtRestCipher::isSealed(const QByteArray &data)
{
    return data.startsWith(kMarker);
}

QByteArray AtRestCipher::seal(const QByteArray &plain, const QString &context)
{
#ifdef Q_OS_WIN
    QByteArray keyMaterial;
    if (!loadKey(&keyMaterial)) { qWarning() << "AtRestCipher: no at-rest key is available; the payload was not sealed."; return {}; }
    const QByteArray nonce = randomBytes(kNonceBytes);
    if (nonce.size() != kNonceBytes) { qWarning() << "AtRestCipher: the CSPRNG is unavailable; the payload was not sealed."; return {}; }
    QByteArray cipher;
    QByteArray tag;
    if (!gcmEncrypt(keyMaterial, nonce, associatedData(context), plain, &cipher, &tag)) { qWarning() << "AtRestCipher: sealing the payload failed."; return {}; }
    return kMarker + nonce + cipher + tag;
#else
    // No AEAD backend on this platform: the payload stays readable for the
    // Python service, which owns the real cryptography.
    Q_UNUSED(context);
    return plain;
#endif
}

QByteArray AtRestCipher::open(const QByteArray &sealed, const QString &context)
{
    if (!isSealed(sealed)) {
        // A payload written before the envelope existed: hand it back
        // unchanged so the old attachments keep rendering.
        return sealed;
    }
#ifdef Q_OS_WIN
    QByteArray keyMaterial;
    if (!loadKey(&keyMaterial)) { qWarning() << "AtRestCipher: no at-rest key is available; the payload could not be unsealed."; return {}; }
    const QByteArray body = sealed.mid(static_cast<int>(sizeof(kMarker) - 1));
    if (body.size() < kNonceBytes + kTagBytes) { qWarning() << "AtRestCipher: the sealed payload is truncated."; return {}; }
    const QByteArray nonce = body.left(kNonceBytes);
    const QByteArray tag = body.right(kTagBytes);
    const QByteArray cipher = body.mid(kNonceBytes, body.size() - kNonceBytes - kTagBytes);
    QByteArray plain;
    if (!gcmDecrypt(keyMaterial, nonce, associatedData(context), cipher, tag, &plain)) { qWarning() << "AtRestCipher: the payload could not be unsealed (wrong key, wrong context or tampered bytes)."; return {}; }
    return plain;
#else
    Q_UNUSED(context);
    return sealed;
#endif
}
