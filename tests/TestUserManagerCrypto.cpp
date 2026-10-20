#include <QtTest>
#include <QTemporaryDir>
#include <QSettings>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>

#include "UserManager.h"
#include "UserAccountStore.h"
#include "SecretVault.h"
#include "IOutgoingMail.h"
#include "CryptoUtils.h"
#include "OtpStore.h"

/**
 * @brief Regression tests for UserManager password hashing and SecretVault.
 *
 * Covers the P0 security contract:
 *  - fresh hashes are salted PBKDF2 (never bare SHA-256),
 *  - legacy plaintext-hash accounts transparently upgrade at login,
 *  - wrong passwords are rejected for both formats,
 *  - SecretVault round-trips and never stores unprotected secrets verbatim.
 */
class FakeOutgoingMail : public IOutgoingMail {
public:
    bool send(const QString &from, const QString &to, const QString &subject,
              const QString &htmlBody, QString &errorMsg) override
    {
        Q_UNUSED(errorMsg)
        lastFrom = from;
        lastTo = to;
        lastSubject = subject;
        lastBody = htmlBody;
        return success;
    }

    bool success = true;
    QString lastFrom;
    QString lastTo;
    QString lastSubject;
    QString lastBody;
};

class TestUserManagerCrypto : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void init();
    void freshAdminHashUsesPbkdf2();
    void generatedAdminPasswordIsWrittenAndUsable();
    void unwritablePasswordFileRefusesKnownFallback();
    void legacyHashIsTransparentlyUpgraded();
    void changePasswordFlowRejectsWrongOldPassword();
    void rememberMeProfileAndPasswordReset();
    void userCrudPreferencesAndSecrets();
    void accountStoreOwnsCredentialMap();
    void otpLifecycleUsesInjectedMail();
    void licenseLifecycleAndKeyLedger();
    void loginErrorCodesAndUnknownUsers();
    void cryptoUtilsHandlesLongKeys();
    void otpStoreExpiryAndClear();
    void secretVaultRoundtrip();
    void secretVaultLegacyPlaintextPassthrough();
    void licenseKeysStoredEncrypted();

private:
    static QString legacySha256Hex(const QString &password) {
        return QString::fromLatin1(
            QCryptographicHash::hash(password.toUtf8(), QCryptographicHash::Sha256).toHex());
    }
};

void TestUserManagerCrypto::initTestCase()
{
    // Production generates a random credential unless the installer provides
    // one; pin the test fixture explicitly instead of relying on a default.
    qputenv("APP_ADMIN_INITIAL_PASSWORD", "1");
}

void TestUserManagerCrypto::init()
{
    // Reset before every test: the bootstrap tests below deliberately unset
    // the variable to exercise the generated-credential path.
    qputenv("APP_ADMIN_INITIAL_PASSWORD", "1");
}

void TestUserManagerCrypto::freshAdminHashUsesPbkdf2()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QString ini = dir.path() + "/users.ini";

    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));
    QVERIFY(UserManager::instance()->login("admin", "1", err));
    QVERIFY(err.isEmpty());

    err.clear();
    // Read the persisted credential: UserManager no longer exposes hashes via
    // UserInfo, so the INI is the observable storage boundary.
    QSettings stored(ini, QSettings::IniFormat);
    const QString hash = stored.value("User_admin/password_hash").toString();
    QVERIFY(hash.startsWith("pbkdf2$100000$"));
    QVERIFY(!UserManager::instance()->login("admin", "wrong-password", err));
    QVERIFY(!err.isEmpty());
    UserManager::instance()->logout();
}


void TestUserManagerCrypto::generatedAdminPasswordIsWrittenAndUsable()
{
    qunsetenv("APP_ADMIN_INITIAL_PASSWORD");
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";

    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));

    const QString passwordFile = dir.path() + "/admin_initial_password.txt";
    QVERIFY(QFileInfo::exists(passwordFile));

    QFile file(passwordFile);
    QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString content = QString::fromUtf8(file.readAll());
    file.close();

    static const QRegularExpression re(
        QStringLiteral("Initial admin password: (\\S+)"));
    const QRegularExpressionMatch match = re.match(content);
    QVERIFY2(match.hasMatch(), qPrintable(content));

    const QString password = match.captured(1);
    QVERIFY(password.size() >= 20);
    QVERIFY(UserManager::instance()->login("admin", password, err));
    QVERIFY(err.isEmpty());
    QVERIFY(UserManager::instance()->isLoggedIn());
    QVERIFY(UserManager::instance()->allUsernames().contains("admin"));
    QCOMPARE(UserManager::instance()->userInfo("admin").role, UserRole::Admin);
    UserManager::instance()->logout();
    QVERIFY(!UserManager::instance()->isLoggedIn());
}

void TestUserManagerCrypto::unwritablePasswordFileRefusesKnownFallback()
{
    qunsetenv("APP_ADMIN_INITIAL_PASSWORD");
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    // A directory at the credential-file path makes QFile::open() fail. The
    // manager must invent an unusable random credential instead of falling
    // back to the historical well-known "1".
    QVERIFY(QDir(dir.path()).mkdir(QStringLiteral("admin_initial_password.txt")));

    const QString ini = dir.path() + "/users.ini";
    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));
    UserManager::instance()->logout();
    QVERIFY(!UserManager::instance()->login("admin", "1", err));
    QVERIFY(!err.isEmpty());
}

void TestUserManagerCrypto::legacyHashIsTransparentlyUpgraded()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QString ini = dir.path() + "/users.ini";
    {
        QSettings seed(ini, QSettings::IniFormat);
        seed.setValue("Users/usernames", QStringList{"legacy"});
        seed.setValue("User_legacy/password_hash", legacySha256Hex("LegacyPass1!"));
        seed.sync();
    }

    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));
    QVERIFY(!UserManager::instance()->login("legacy", "wrong1", err));
    QVERIFY(!err.isEmpty());

    err.clear();
    QVERIFY(UserManager::instance()->login("legacy", "LegacyPass1!", err));
    QSettings stored(ini, QSettings::IniFormat);
    QVERIFY(stored.value("User_legacy/password_hash").toString().startsWith("pbkdf2$"));
}

void TestUserManagerCrypto::changePasswordFlowRejectsWrongOldPassword()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QString ini = dir.path() + "/users.ini";
    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));

    UserInfo info;
    info.username = "p1";
    info.email    = "p1@example.com";
    info.role     = UserRole::User;
    QVERIFY(UserManager::instance()->addUser(info, "Secret1!", err));

    err.clear();
    QVERIFY(!UserManager::instance()->changePassword("p1", "not-the-old", "NewPass1!", err));
    QVERIFY(!err.isEmpty());

    err.clear();
    QVERIFY(UserManager::instance()->changePassword("p1", "Secret1!", "NewPass1!", err));
    QVERIFY(err.isEmpty());

    err.clear();
    QVERIFY(!UserManager::instance()->login("p1", "Secret1!", err));
    err.clear();
    QVERIFY(UserManager::instance()->login("p1", "NewPass1!", err));
}

void TestUserManagerCrypto::rememberMeProfileAndPasswordReset()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";

    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));
    QVERIFY(UserManager::instance()->login("admin", "1", err));
    QCOMPARE(UserManager::instance()->currentUsername(), QStringLiteral("admin"));
    QCOMPARE(UserManager::instance()->currentUser().username, QStringLiteral("admin"));

    UserManager::instance()->setRememberMe(true, "admin");
    QVERIFY(UserManager::instance()->isRememberMeEnabled());
    QCOMPARE(UserManager::instance()->savedUsername(), QStringLiteral("admin"));
    UserManager::instance()->setRememberMe(false, "admin");
    QVERIFY(!UserManager::instance()->isRememberMeEnabled());
    QVERIFY(UserManager::instance()->savedUsername().isEmpty());

    UserManager::instance()->updateAvatar("admin", QStringLiteral("/tmp/avatar.png"));
    QCOMPARE(UserManager::instance()->userInfo("admin").avatarPath,
             QStringLiteral("/tmp/avatar.png"));
    UserManager::instance()->updateAvatar("missing", QStringLiteral("/tmp/none.png"));

    QVERIFY(!UserManager::instance()->resetPassword("missing", "NewPass1!"));
    QVERIFY(UserManager::instance()->resetPassword("admin", "NewPass1!"));
    UserManager::instance()->logout();
    QVERIFY(!UserManager::instance()->isLoggedIn());

    err.clear();
    QVERIFY(!UserManager::instance()->login("admin", "1", err));
    QVERIFY(!err.isEmpty());
    err.clear();
    QVERIFY(UserManager::instance()->login("admin", "NewPass1!", err));
    QVERIFY(err.isEmpty());
}

void TestUserManagerCrypto::userCrudPreferencesAndSecrets()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";
    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));

    UserInfo user;
    user.username = "p1";
    user.email = "p1@example.com";
    user.role = UserRole::User;

    AuthError code = AuthError::None;
    QVERIFY(!UserManager::instance()->addUser(user, "123", err, &code));
    QVERIFY(code == AuthError::WeakPassword);
    err.clear();
    code = AuthError::None;
    QVERIFY(UserManager::instance()->addUser(user, "Secret1!", err, &code));
    QVERIFY(code == AuthError::None);
    QVERIFY(UserManager::instance()->allUsernames().contains("p1"));
    QCOMPARE(UserManager::instance()->userInfo("p1").email, QStringLiteral("p1@example.com"));

    err.clear();
    code = AuthError::None;
    QVERIFY(!UserManager::instance()->addUser(user, "Secret1!", err, &code));
    QVERIFY(code == AuthError::UserExists);

    UserInfo missing;
    missing.username = "missing";
    QVERIFY(!UserManager::instance()->updateUser(missing, err));
    user.email = "p1-new@example.com";
    QVERIFY(UserManager::instance()->updateUser(user, err));
    QCOMPARE(UserManager::instance()->userInfo("p1").email, QStringLiteral("p1-new@example.com"));

    err.clear();
    QVERIFY(UserManager::instance()->login("p1", "Secret1!", err));
    QVERIFY(UserManager::instance()->isLoggedIn());
    QVERIFY(!UserManager::instance()->deleteUser("p1", err)); // current user
    UserManager::instance()->logout();
    QVERIFY(!UserManager::instance()->deleteUser("admin", err)); // protected account
    QVERIFY(!UserManager::instance()->deleteUser("missing", err));
    QVERIFY(UserManager::instance()->deleteUser("p1", err));
    QVERIFY(!UserManager::instance()->allUsernames().contains("p1"));

    UserManager::instance()->setUserPref("admin", "theme", "dark");
    QCOMPARE(UserManager::instance()->getUserPref("admin", "theme", "light"),
             QStringLiteral("dark"));
    UserManager::instance()->setUserPref("", "theme", "dark");
    QCOMPARE(UserManager::instance()->getUserPref("", "theme", "fallback"),
             QStringLiteral("fallback"));
    QCOMPARE(UserManager::instance()->getUserPref("admin", "missing", "fallback"),
             QStringLiteral("fallback"));

    UserManager::instance()->setUserSecret("admin", "mail_password", "s3cret");
    QCOMPARE(UserManager::instance()->getUserSecret("admin", "mail_password", ""),
             QStringLiteral("s3cret"));
    QCOMPARE(UserManager::instance()->getUserSecret("admin", "missing", "fallback"),
             QStringLiteral("fallback"));
    UserManager::instance()->setUserSecret("", "mail_password", "ignored");
    QCOMPARE(UserManager::instance()->getUserSecret("", "mail_password", "fallback"),
             QStringLiteral("fallback"));
}

void TestUserManagerCrypto::accountStoreOwnsCredentialMap()
{
    // UserManager is a facade; the store is the persistence boundary that
    // keeps credentials out of UserInfo. Exercise it directly so the
    // credential map can never be removed or bypassed silently.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QSettings settings(dir.path() + "/store.ini", QSettings::IniFormat);
    UserAccountStore store;
    store.attachSettings(&settings);

    UserInfo info;
    info.username = QStringLiteral("store-user");
    info.email = QStringLiteral("store@example.com");
    info.role = UserRole::User;
    store.insert(info, QStringLiteral("hash-1"));

    QVERIFY(store.contains(QStringLiteral("store-user")));
    QCOMPARE(store.info(QStringLiteral("store-user")).email,
             QStringLiteral("store@example.com"));
    QCOMPARE(store.usernames(), QStringList{QStringLiteral("store-user")});
    QCOMPARE(store.passwordHash(QStringLiteral("store-user")),
             QStringLiteral("hash-1"));

    store.setPasswordHash(QStringLiteral("store-user"), QStringLiteral("hash-2"));
    QCOMPARE(store.passwordHash(QStringLiteral("store-user")),
             QStringLiteral("hash-2"));

    info.email = QStringLiteral("new@example.com");
    store.update(info);
    QCOMPARE(store.info(QStringLiteral("store-user")).email,
             QStringLiteral("new@example.com"));

    QCOMPARE(store.findLoginUsername(QStringLiteral("STORE-USER")),
             QStringLiteral("store-user"));
    QCOMPARE(store.findLoginUsername(QStringLiteral("NEW@EXAMPLE.COM")),
             QStringLiteral("store-user"));
    QCOMPARE(store.findUsernameByEmail(QStringLiteral("new@example.com")),
             QStringLiteral("store-user"));

    store.remove(QStringLiteral("store-user"));
    QVERIFY(!store.contains(QStringLiteral("store-user")));
    QVERIFY(store.passwordHash(QStringLiteral("store-user")).isEmpty());
}

void TestUserManagerCrypto::otpLifecycleUsesInjectedMail()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";
    QString err;
    UserManager::instance()->setOutgoingMail(nullptr);
    QVERIFY(UserManager::instance()->loadConfig(ini));

    QCOMPARE(UserManager::instance()->findUsernameByEmail("ADMIN@EXAMPLE.COM"),
             QStringLiteral("admin"));
    QVERIFY(UserManager::instance()->findUsernameByEmail("missing@example.com").isEmpty());
    QVERIFY(!UserManager::instance()->sendPasswordResetOtp("missing@example.com", err));
    QVERIFY(!err.isEmpty());
    err.clear();
    QVERIFY(!UserManager::instance()->sendPasswordResetOtp("admin@example.com", err));
    QVERIFY(!err.isEmpty());

    FakeOutgoingMail mail;
    UserManager::instance()->setOutgoingMail(&mail);
    err.clear();
    QVERIFY(!UserManager::instance()->sendPasswordResetOtp("admin@example.com", err));
    QVERIFY(err.contains(QStringLiteral("Chưa cấu hình")));

    UserManager::instance()->setSmtpCredentials("sender@example.com", "smtp-secret");
    QCOMPARE(UserManager::instance()->smtpSenderEmail(), QStringLiteral("sender@example.com"));
    QCOMPARE(UserManager::instance()->smtpSenderPassword(), QStringLiteral("smtp-secret"));

    err.clear();
    QVERIFY(UserManager::instance()->sendPasswordResetOtp("admin@example.com", err));
    QVERIFY(err.isEmpty());
    QCOMPARE(mail.lastFrom, QStringLiteral("sender@example.com"));
    QCOMPARE(mail.lastTo, QStringLiteral("admin@example.com"));
    QVERIFY(mail.lastBody.contains(QStringLiteral("Mã xác nhận")));

    const QRegularExpressionMatch match =
        QRegularExpression(QStringLiteral("<b>(\\d{6})</b>")).match(mail.lastBody);
    QVERIFY2(match.hasMatch(), qPrintable(mail.lastBody));
    const QString otp = match.captured(1);

    QVERIFY(!UserManager::instance()->verifyOtp("admin@example.com", "000000"));
    QVERIFY(UserManager::instance()->verifyOtp("admin@example.com", otp));
    QVERIFY(!UserManager::instance()->verifyOtp("admin@example.com", otp)); // one-shot
    QVERIFY(!UserManager::instance()->verifyOtp("missing@example.com", otp));
    UserManager::instance()->setOutgoingMail(nullptr);
}

void TestUserManagerCrypto::licenseLifecycleAndKeyLedger()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";
    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));

    UserInfo user;
    user.username = "licensed";
    user.email = "licensed@example.com";
    QVERIFY(UserManager::instance()->addUser(user, "Secret1!", err));

    AuthError code = AuthError::None;
    QVERIFY(!UserManager::instance()->activateTrial("missing", err, &code));
    QVERIFY(code == AuthError::UserNotFound);
    QVERIFY(UserManager::instance()->needsActivation("licensed"));
    QCOMPARE(UserManager::instance()->licenseStatusStr("licensed"), QStringLiteral("None"));
    QVERIFY(UserManager::instance()->isLicenseExpired("licensed"));
    QCOMPARE(UserManager::instance()->daysRemaining("licensed"), 0);
    QVERIFY(!UserManager::instance()->needsActivation("missing"));
    QVERIFY(UserManager::instance()->isLicenseExpired("missing"));
    QCOMPARE(UserManager::instance()->daysRemaining("missing"), 0);

    err.clear();
    QVERIFY(UserManager::instance()->activateTrial("licensed", err, &code));
    QVERIFY(!UserManager::instance()->needsActivation("licensed"));
    QVERIFY(!UserManager::instance()->isLicenseExpired("licensed"));
    QVERIFY(UserManager::instance()->daysRemaining("licensed") > 0);
    QCOMPARE(UserManager::instance()->licenseStatusStr("licensed"), QStringLiteral("Trial"));
    QVERIFY(!UserManager::instance()->activateTrial("licensed", err, &code));
    QVERIFY(code == AuthError::AlreadyActivated);

    const QString key = UserManager::instance()->generateLicenseKey();
    QVERIFY(!key.isEmpty());
    QVERIFY(UserManager::instance()->availableLicenseKeys().contains(key));
    UserManager::instance()->saveConfig();

    // Persisted key inventory survives a reload.
    QVERIFY(UserManager::instance()->loadConfig(ini));
    QVERIFY(UserManager::instance()->availableLicenseKeys().contains(key));

    QVERIFY(!UserManager::instance()->activateLicenseKey("missing", key, err, &code));
    QVERIFY(code == AuthError::UserNotFound);
    QVERIFY(!UserManager::instance()->activateLicenseKey("licensed", "bad-key", err, &code));
    QVERIFY(code == AuthError::InvalidKey);
    QVERIFY(UserManager::instance()->activateLicenseKey("licensed", key, err, &code));
    QCOMPARE(UserManager::instance()->licenseStatusStr("licensed"), QStringLiteral("Activated"));

    UserInfo second;
    second.username = "second";
    second.email = "second@example.com";
    QVERIFY(UserManager::instance()->addUser(second, "Secret1!", err));
    QVERIFY(!UserManager::instance()->activateLicenseKey("second", key, err, &code));
    QVERIFY(code == AuthError::KeyAlreadyUsed);

    QVERIFY(!UserManager::instance()->revokeLicenseKey("missing-key", err));
    QVERIFY(UserManager::instance()->revokeLicenseKey(key, err));
    QCOMPARE(UserManager::instance()->licenseStatusStr("licensed"), QStringLiteral("None"));
    QVERIFY(!UserManager::instance()->availableLicenseKeys().contains(key));
    QVERIFY(!UserManager::instance()->revokeLicenseKey(key, err));
}

void TestUserManagerCrypto::loginErrorCodesAndUnknownUsers()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";
    QString err;
    QVERIFY(UserManager::instance()->loadConfig(ini));

    AuthError code = AuthError::None;
    QVERIFY(!UserManager::instance()->login("missing", "whatever", err, &code));
    QVERIFY(code == AuthError::UserNotFound);
    QCOMPARE(IUserManager::authErrorKey(code), QStringLiteral("auth.no_user"));

    err.clear();
    code = AuthError::None;
    QVERIFY(!UserManager::instance()->login("admin", "wrong", err, &code));
    QVERIFY(code == AuthError::WrongPassword);

    err.clear();
    code = AuthError::None;
    QVERIFY(!UserManager::instance()->changePassword("missing", "1", "NewPass1!", err, &code));
    QVERIFY(code == AuthError::UserNotFound);
    QVERIFY(!UserManager::instance()->changePassword("admin", "wrong", "NewPass1!", err, &code));
    QVERIFY(code == AuthError::WrongPassword);
    QVERIFY(!UserManager::instance()->changePassword("admin", "1", "123", err, &code));
    QVERIFY(code == AuthError::WeakPassword);
    QVERIFY(UserManager::instance()->changePassword("admin", "1", "NewPass1!", err, &code));

    err.clear();
    QVERIFY(!UserManager::instance()->login("admin", "1", err));
    err.clear();
    QVERIFY(UserManager::instance()->login("admin@example.com", "NewPass1!", err));
}

void TestUserManagerCrypto::cryptoUtilsHandlesLongKeys()
{
    // A password longer than SHA-256's 64-byte block exercises the HMAC key
    // pre-hash path (the usual short-password tests never reach it).
    const QString password = QString(80, QLatin1Char('a'));
    const QString hash = CryptoUtils::hashPassword(password);
    QVERIFY(CryptoUtils::isModernPasswordHash(hash));
    QVERIFY(CryptoUtils::verifyPassword(password, hash));
    QVERIFY(!CryptoUtils::verifyPassword(password + QStringLiteral("x"), hash));
    QVERIFY(!CryptoUtils::verifyPassword(password, QStringLiteral("pbkdf2$broken")));
    QVERIFY(!CryptoUtils::verifyPassword(password, QStringLiteral("pbkdf2$1$00$00")));
    QVERIFY(CryptoUtils::constantTimeEquals(QByteArray("same"), QByteArray("same")));
    QVERIFY(!CryptoUtils::constantTimeEquals(QByteArray("short"), QByteArray("longer")));
    QCOMPARE(CryptoUtils::sha256Hex(QStringLiteral("abc")),
             legacySha256Hex(QStringLiteral("abc")));
}

void TestUserManagerCrypto::otpStoreExpiryAndClear()
{
    OtpStore store;
    const QString expired = store.issue(QStringLiteral("USER@Example.com"), -1);
    QVERIFY(!store.verify(QStringLiteral("user@example.com"), expired));

    const QString live = store.issue(QStringLiteral("USER@Example.com"));
    QVERIFY(store.verify(QStringLiteral("user@example.com"), live));
    QVERIFY(!store.verify(QStringLiteral("user@example.com"), live));

    const QString cleared = store.issue(QStringLiteral("x@example.com"));
    store.clear();
    QVERIFY(!store.verify(QStringLiteral("x@example.com"), cleared));
}

void TestUserManagerCrypto::secretVaultRoundtrip()
{
    const QString secret = "app-password-示例-🔐";
    const QString stored = SecretVault::encrypt(secret);
    // Never store the raw secret verbatim.
    QVERIFY(!stored.contains(secret));
    QVERIFY(stored.startsWith("enc1:") || stored.startsWith("plain:"));
    QCOMPARE(SecretVault::decrypt(stored), secret);
    QVERIFY(SecretVault::decrypt("").isEmpty());
    // Empty stays empty: an "encrypted" blob for "" would later decrypt into
    // a non-empty string (or worse, a CryptProtectData round-trip of nothing).
    QVERIFY(SecretVault::encrypt(QString()).isEmpty());
    // A blob that is not DPAPI-protected (foreign user/machine, corrupted
    // base64 payload) must decrypt to empty, never surface garbage as a
    // plaintext secret.
    QVERIFY(SecretVault::decrypt(QStringLiteral("enc1:bm90LWEtdmFsaWQtYmxvYg==")).isEmpty());
}

void TestUserManagerCrypto::secretVaultLegacyPlaintextPassthrough()
{
    // Values written before the vault existed are passed through so old
    // Config.ini files keep working until the next save migrates them.
    QCOMPARE(SecretVault::decrypt("old-plaintext"), QLatin1String("old-plaintext"));
    QCOMPARE(SecretVault::decrypt("plain:abc"), QLatin1String("abc"));
    QVERIFY(SecretVault::decrypt("").isEmpty());
}

void TestUserManagerCrypto::licenseKeysStoredEncrypted()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString ini = dir.path() + "/users.ini";

    // Seed the legacy plaintext format: key list + plaintext key in paths.
    {
        QSettings seed(ini, QSettings::IniFormat);
        seed.setValue("LicenseKeys/keys", QStringList{"LEGACY-KEY-XYZ"});
        seed.setValue("LicenseKeys/LEGACY-KEY-XYZ_used", true);
        seed.setValue("LicenseKeys/LEGACY-KEY-XYZ_user", "admin");
        seed.sync();
    }

    QVERIFY(UserManager::instance()->loadConfig(ini));

    // The legacy key survives the migration in memory...
    QVERIFY(UserManager::instance()->availableLicenseKeys().contains("LEGACY-KEY-XYZ"));

    // ...and new keys work through the normal generation flow.
    const QString fresh = UserManager::instance()->generateLicenseKey();
    QVERIFY(!fresh.isEmpty());
    QVERIFY(UserManager::instance()->availableLicenseKeys().contains(fresh));
    UserManager::instance()->saveConfig();

    // The persisted INI must contain no plaintext keys — values are DPAPI
    // blobs and paths are hashed.
    QFile f(ini);
    QVERIFY(f.open(QIODevice::ReadOnly));
    const QString content = QString::fromUtf8(f.readAll());
    f.close();
    QVERIFY(!content.contains("LEGACY-KEY-XYZ"));
    QVERIFY(!content.contains(fresh));
    QVERIFY(content.contains("enc1:"));
}

QTEST_MAIN(TestUserManagerCrypto)
#include "TestUserManagerCrypto.moc"
