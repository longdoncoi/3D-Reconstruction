#include <QtTest>

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include "AtRestCipher.h"

/**
 * @brief Contract tests for the at-rest cipher (src/core/AtRestCipher.cpp).
 *
 * The envelope is a shared contract with the Python service
 * (ai_assistant.adapters.persistence.storage_crypto): the byte layout, the
 * associated data and the key resolution must match exactly, otherwise the
 * Python side cannot read what this side writes.
 */

namespace {

const char kAttachmentContext[] = "chat.attachment";
const char kThumbnailContext[] = "chat.thumbnail";

QByteArray testKey(char fill)
{
    return QByteArray(32, fill);
}

} // namespace

class TestAtRestCipher : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();

    void generatedKeyFileIsCreatedAndReused();
    void sealProducesTheDocumentedEnvelope();
    void sealAndOpenRoundTrip();
    void twoSealsUseDifferentNonces();
    void wrongContextIsRejected();
    void tamperedCiphertextIsRejected();
    void truncatedEnvelopeIsRejected();
    void nonSealedPayloadPassesThrough();
    void keyFromEnvironmentIsAccepted();
    void malformedEnvironmentKeyFailsClosed();
    void keyFromEnvironmentFileIsAccepted();
    void missingKeyFileFailsClosed();
    void unusableConfigurationDirectoryFailsClosed();
    void platformWithoutAeadKeepsThePayloadReadable();
    void thePythonServiceEnvelopeIsReadable();
    void dumpInteropEnvelope();

private:
    /// Temporary stand-in for the user configuration directory.
    QTemporaryDir m_home;
    /// The configuration directory the environment pointed at before.
    QString m_oldConfigHome;
};

void TestAtRestCipher::initTestCase()
{
    QVERIFY(m_home.isValid());
    // The generated key must never land in the real user profile: the whole
    // test runs against a temporary configuration directory.
#ifdef Q_OS_WIN
    m_oldConfigHome = qEnvironmentVariable("LOCALAPPDATA");
    qputenv("LOCALAPPDATA", m_home.path().toLocal8Bit());
#else
    m_oldConfigHome = qEnvironmentVariable("XDG_CONFIG_HOME");
    qputenv("XDG_CONFIG_HOME", m_home.path().toLocal8Bit());
#endif
    qunsetenv("AI_STORAGE_KEY");
    qunsetenv("AI_STORAGE_KEY_FILE");
}

void TestAtRestCipher::cleanupTestCase()
{
    qunsetenv("AI_STORAGE_KEY");
    qunsetenv("AI_STORAGE_KEY_FILE");
#ifdef Q_OS_WIN
    if (m_oldConfigHome.isEmpty()) {
        qunsetenv("LOCALAPPDATA");
    } else {
        qputenv("LOCALAPPDATA", m_oldConfigHome.toLocal8Bit());
    }
#else
    if (m_oldConfigHome.isEmpty()) {
        qunsetenv("XDG_CONFIG_HOME");
    } else {
        qputenv("XDG_CONFIG_HOME", m_oldConfigHome.toLocal8Bit());
    }
#endif
}

void TestAtRestCipher::generatedKeyFileIsCreatedAndReused()
{
    const QString keyPath = QDir(m_home.path()).filePath(QStringLiteral("3d-reconstruction/ai_assistant.key"));
    QFile::remove(keyPath);
    QVERIFY(!QFile::exists(keyPath));

    const QByteArray envelope = AtRestCipher::seal(QByteArray("payload"),
                                                   QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    QVERIFY(QFile::exists(keyPath));

    QFile keyFile(keyPath);
    QVERIFY(keyFile.open(QIODevice::ReadOnly));
    const QByteArray keyText = keyFile.readAll();
    keyFile.close();
    // "base64:<url-safe base64>\n" — the exact layout the Python service
    // writes, so both sides read the same key file.
    QVERIFY(keyText.startsWith("base64:"));
    QVERIFY(keyText.endsWith("\n"));
    const QByteArray encoded = keyText.mid(7, keyText.size() - 8);
    QCOMPARE(QByteArray::fromBase64(encoded, QByteArray::Base64UrlEncoding).size(), 32);

    // The key is reused, never regenerated: the envelope sealed above still
    // opens after another seal.
    const QByteArray second = AtRestCipher::seal(QByteArray("payload"),
                                                  QString::fromLatin1(kAttachmentContext));
    QVERIFY(!second.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)),
             QByteArray("payload"));
    QCOMPARE(AtRestCipher::open(second, QString::fromLatin1(kAttachmentContext)),
             QByteArray("payload"));
}

void TestAtRestCipher::sealProducesTheDocumentedEnvelope()
{
    const QByteArray plain("attachment payload");
    const QByteArray envelope = AtRestCipher::seal(plain, QString::fromLatin1(kAttachmentContext));

    QVERIFY(!envelope.isEmpty());
    QVERIFY(AtRestCipher::isSealed(envelope));
    QVERIFY(envelope.startsWith("enc:v1:"));
    QCOMPARE(AtRestCipher::marker(), QByteArray("enc:v1:"));
    // marker(7) + nonce(12) + ciphertext(len(plaintext)) + tag(16)
    QCOMPARE(envelope.size(), 7 + 12 + plain.size() + 16);
}

void TestAtRestCipher::sealAndOpenRoundTrip()
{
    const QByteArray plain("hello attachment");
    const QByteArray envelope = AtRestCipher::seal(plain, QString::fromLatin1(kAttachmentContext));

    QVERIFY(!envelope.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)), plain);
}

void TestAtRestCipher::twoSealsUseDifferentNonces()
{
    const QByteArray plain("same payload");
    const QByteArray first = AtRestCipher::seal(plain, QString::fromLatin1(kAttachmentContext));
    const QByteArray second = AtRestCipher::seal(plain, QString::fromLatin1(kAttachmentContext));

    QVERIFY(!first.isEmpty());
    QVERIFY(!second.isEmpty());
    QVERIFY(first != second);
    // The nonce is fresh for every envelope.
    QVERIFY(first.mid(7, 12) != second.mid(7, 12));
}

void TestAtRestCipher::wrongContextIsRejected()
{
    const QByteArray envelope = AtRestCipher::seal(QByteArray("payload"),
                                                   QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    // The context is bound into the associated data: a thumbnail must not be
    // replayed as an original attachment.
    QVERIFY(AtRestCipher::open(envelope, QString::fromLatin1(kThumbnailContext)).isEmpty());
}

void TestAtRestCipher::tamperedCiphertextIsRejected()
{
    QByteArray envelope = AtRestCipher::seal(QByteArray("tamper me"),
                                              QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    const int index = envelope.size() - 20;
    envelope[index] = static_cast<char>(envelope.at(index) ^ 0x01);
    QVERIFY(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)).isEmpty());
}

void TestAtRestCipher::truncatedEnvelopeIsRejected()
{
    QByteArray envelope = AtRestCipher::seal(QByteArray("payload"),
                                              QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    envelope.truncate(10);
    QVERIFY(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)).isEmpty());
}

void TestAtRestCipher::nonSealedPayloadPassesThrough()
{
    // An attachment written before the envelope existed stays readable.
    const QByteArray legacy("legacy plaintext attachment");
    QVERIFY(!AtRestCipher::isSealed(legacy));
    QCOMPARE(AtRestCipher::open(legacy, QString::fromLatin1(kAttachmentContext)), legacy);
}

void TestAtRestCipher::keyFromEnvironmentIsAccepted()
{
    const QByteArray key = testKey('k');

    qputenv("AI_STORAGE_KEY", "base64:" + key.toBase64(QByteArray::Base64UrlEncoding));
    QByteArray envelope = AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)), QByteArray("payload"));

    qputenv("AI_STORAGE_KEY", "hex:" + key.toHex());
    envelope = AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)), QByteArray("payload"));

    qputenv("AI_STORAGE_KEY", key.toBase64(QByteArray::Base64UrlEncoding));
    envelope = AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)), QByteArray("payload"));

    // A key whose URL-safe base64 uses the '-' / '_' alphabet.
    const QByteArray urlSafeKey = testKey(static_cast<char>(0xfb));
    QVERIFY(urlSafeKey.toBase64(QByteArray::Base64UrlEncoding).contains('-')
            || urlSafeKey.toBase64(QByteArray::Base64UrlEncoding).contains('_'));
    qputenv("AI_STORAGE_KEY", urlSafeKey.toBase64(QByteArray::Base64UrlEncoding));
    envelope = AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)), QByteArray("payload"));

    qunsetenv("AI_STORAGE_KEY");
}

void TestAtRestCipher::malformedEnvironmentKeyFailsClosed()
{
    qputenv("AI_STORAGE_KEY", QByteArray("not-a-key"));
    // A malformed explicit key never falls back to a key file: the payload
    // is not sealed at all.
    QVERIFY(AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext)).isEmpty());

    const QByteArray envelope = QByteArray("enc:v1:0123456789abcdef0123456789abcdef0123456789abcdef");
    QVERIFY(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)).isEmpty());

    qunsetenv("AI_STORAGE_KEY");
}

void TestAtRestCipher::keyFromEnvironmentFileIsAccepted()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    const QString keyPath = tmp.filePath(QStringLiteral("ai_assistant.key"));
    const QByteArray key = testKey('f');

    QFile file(keyPath);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("base64:" + key.toBase64(QByteArray::Base64UrlEncoding) + "\n");
    file.close();

    qputenv("AI_STORAGE_KEY_FILE", keyPath.toLocal8Bit());
    const QByteArray envelope = AtRestCipher::seal(QByteArray("payload"),
                                                   QString::fromLatin1(kAttachmentContext));
    QVERIFY(!envelope.isEmpty());
    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)), QByteArray("payload"));

    qunsetenv("AI_STORAGE_KEY_FILE");
}

void TestAtRestCipher::missingKeyFileFailsClosed()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    qputenv("AI_STORAGE_KEY_FILE", tmp.filePath(QStringLiteral("missing.key")).toLocal8Bit());

    // A configured key file that does not exist must not be replaced by a
    // generated key: the payload is not sealed at all.
    QVERIFY(AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext)).isEmpty());

    // A "~" path is expanded against the home directory — and still fails
    // closed, because the file does not exist.
    qputenv("AI_STORAGE_KEY_FILE", QByteArray("~/missing-at-rest.key"));
    QVERIFY(AtRestCipher::seal(QByteArray("payload"), QString::fromLatin1(kAttachmentContext)).isEmpty());

    qunsetenv("AI_STORAGE_KEY_FILE");
}

void TestAtRestCipher::unusableConfigurationDirectoryFailsClosed()
{
    QTemporaryDir tmp;
    QVERIFY(tmp.isValid());
    // A regular file where the configuration directory should be: the key
    // file can neither be created nor read, so nothing may be sealed.
    const QString blocker = tmp.filePath(QStringLiteral("blocker"));
    QFile file(blocker);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("not a directory");
    file.close();

#ifdef Q_OS_WIN
    const QByteArray saved = qEnvironmentVariable("LOCALAPPDATA").toLocal8Bit();
    qputenv("LOCALAPPDATA", blocker.toLocal8Bit());
#else
    const QByteArray saved = qEnvironmentVariable("XDG_CONFIG_HOME").toLocal8Bit();
    qputenv("XDG_CONFIG_HOME", blocker.toLocal8Bit());
#endif
    const QByteArray envelope = AtRestCipher::seal(QByteArray("payload"),
                                                   QString::fromLatin1(kAttachmentContext));
    if (saved.isEmpty()) {
#ifdef Q_OS_WIN
        qunsetenv("LOCALAPPDATA");
#else
        qunsetenv("XDG_CONFIG_HOME");
#endif
    } else {
#ifdef Q_OS_WIN
        qputenv("LOCALAPPDATA", saved);
#else
        qputenv("XDG_CONFIG_HOME", saved);
#endif
    }

    QVERIFY(envelope.isEmpty());
}

void TestAtRestCipher::platformWithoutAeadKeepsThePayloadReadable()
{
#ifndef Q_OS_WIN
    // Without a platform AEAD the payload stays readable: the Python service
    // owns the real cryptography.
    const QByteArray plain("payload");
    QCOMPARE(AtRestCipher::seal(plain, QString::fromLatin1(kAttachmentContext)), plain);
    QCOMPARE(AtRestCipher::open(plain, QString::fromLatin1(kAttachmentContext)), plain);
#else
    QVERIFY(AtRestCipher::isSealed(AtRestCipher::seal(QByteArray("payload"),
                                                       QString::fromLatin1(kAttachmentContext))));
#endif
}

void TestAtRestCipher::thePythonServiceEnvelopeIsReadable()
{
    // The cross-language contract: the Python service
    // (ai_assistant.adapters.persistence.storage_crypto) seals an envelope
    // and this side has to open it. The vector is fixed here so a change to
    // either implementation that breaks the other fails here instead of
    // surfacing as "the assistant cannot see the image".
    //
    // Vector: key = 32 x 0x5A, context = "chat.attachment",
    //         plaintext = "ai-assistant at-rest interop vector"
    const QByteArray key = QByteArray(32, char(0x5A));
    qputenv("AI_STORAGE_KEY", key.toBase64(QByteArray::Base64UrlEncoding));
    const QByteArray envelope = QByteArray::fromHex(
        "656e633a76313a77906afb38dfe822faff2836539a6e0aaf5cd33c1d4e767285de382c20d0f64976c3"
        "da8ca9ff527e5f61979b50f6fa9b8bda44fbc00cd1243740c6d5438b43");

    QCOMPARE(AtRestCipher::open(envelope, QString::fromLatin1(kAttachmentContext)),
             QByteArray("ai-assistant at-rest interop vector"));
}

void TestAtRestCipher::dumpInteropEnvelope()
{
    // Not an assertion: a helper that prints a fresh envelope for the
    // cross-language vector, so the Python side can be re-pinned whenever the
    // format changes. Run it and paste the hex into
    // AIAssistant/tests/unit/test_sealed_attachment_read.py.
    const QByteArray key = QByteArray(32, char(0x5A));
    qputenv("AI_STORAGE_KEY", key.toBase64(QByteArray::Base64UrlEncoding));
    const QByteArray envelope = AtRestCipher::seal(
        QByteArray("ai-assistant at-rest interop vector"),
        QString::fromLatin1(kAttachmentContext));
    qInfo("INTEROP-ENVELOPE-HEX %s", envelope.toHex().constData());
}

QTEST_GUILESS_MAIN(TestAtRestCipher)
#include "TestAtRestCipher.moc"
