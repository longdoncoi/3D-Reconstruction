#ifndef ATRESTCIPHER_H
#define ATRESTCIPHER_H

#include <QByteArray>
#include <QString>
#include "Global.h"

/**
 * @brief AES-256-GCM envelope for bytes that must not rest on disk in the clear.
 *
 * The envelope is byte-for-byte the one the Python service writes
 * (ai_assistant.adapters.persistence.storage_crypto.AesGcmFieldCipher):
 *
 *   "enc:v1:"   7 ASCII bytes — the version marker
 *   nonce       12 random bytes from the OS CSPRNG, never reused
 *   ciphertext  len(plaintext) bytes
 *   tag         16 bytes — the GCM authentication tag
 *
 * The associated data is the UTF-8 bytes of
 * "ai_assistant.local_state.v1:" + context, so a payload that is moved to
 * another context (a thumbnail replayed as an original attachment) fails
 * authentication instead of decrypting into something else.
 *
 * The key is resolved exactly like the Python side resolves it
 * (ai_assistant.settings.storage_key / storage_key_location):
 *
 *   1. AI_STORAGE_KEY — "base64:<url-safe base64>", "hex:<hex digits>" or a
 *      bare base64 value. It must decode to exactly 32 bytes, otherwise the
 *      operation fails closed — an explicit key never falls back to a file.
 *   2. AI_STORAGE_KEY_FILE — a file holding "base64:<base64>" or a bare
 *      base64 value. A missing or malformed file fails closed.
 *   3. Otherwise one key file outside the data folder:
 *      %LOCALAPPDATA%\3d-reconstruction\ai_assistant.key (Windows, falling
 *      back to %APPDATA% and then to ~/AppData/Local) or
 *      $XDG_CONFIG_HOME/3d-reconstruction/ai_assistant.key (POSIX, falling
 *      back to ~/.config). It is created exactly once (exclusive create,
 *      owner-only) and reused from then on.
 *
 * Without a platform AEAD backend (a non-Windows build) seal() returns the
 * plaintext unchanged and open() returns its input unchanged: the payload
 * stays readable for the Python service, which owns the real cryptography.
 */
class APP_EXPORT AtRestCipher {
public:
    /// The envelope marker ("enc:v1:").
    static QByteArray marker();

    /// Whether the bytes carry the sealed-envelope marker.
    static bool isSealed(const QByteArray &data);

    /// Seal the plaintext for the given context. Returns an empty array when
    /// the key is unavailable or the platform has no AEAD backend — the
    /// caller must not fall back to storing the plaintext.
    static QByteArray seal(const QByteArray &plain, const QString &context);

    /// Unseal an envelope. A payload without the marker is returned unchanged
    /// (a legacy plaintext file written before the envelope existed); a
    /// sealed payload that cannot be authenticated returns an empty array.
    static QByteArray open(const QByteArray &sealed, const QString &context);
};

#endif // ATRESTCIPHER_H
