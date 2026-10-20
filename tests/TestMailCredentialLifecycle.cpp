#include <QtTest>
#include <QRegularExpression>
#include <QTemporaryDir>

#include "UserManager.h"
#include "MailService.h"
#include "MailThreadGuard.h"
#include "IUserManager.h"

/**
 * @brief Regression tests for MailService credential caching.
 *
 * Covers the P0 credential-lifetime contract:
 *  - credentials are cached for the logged-in user and visible to every
 *    MailService instance (the cache is a derived value of user state),
 *  - logout invalidates the cache — a previous session's account is never
 *    reused after the user changes state,
 *  - with no logged-in user the service reports no credentials.
 *
 * Uses the real UserManager singleton with a throwaway INI (same pattern as
 * TestUserManagerCrypto) — no network is touched.
 */
class TestMailCredentialLifecycle : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void noCredentialsWithoutLogin();
    void credentialsCachedForCurrentUser();
    void credentialsInvalidatedOnLogout();
    void blockingMailGuardWarnsOnGuiThread();
};

void TestMailCredentialLifecycle::initTestCase()
{
    // See TestUserManagerCrypto: the default admin credential is provisioned
    // explicitly in tests, never hard-coded in production code.
    qputenv("APP_ADMIN_INITIAL_PASSWORD", "1");
}

void TestMailCredentialLifecycle::blockingMailGuardWarnsOnGuiThread()
{
#ifdef QT_DEBUG
    // The guard asserts in debug builds (Q_ASSERT_X), which aborts the test
    // process — the warn path is pinned in Release, where CI runs it.
    QSKIP("Q_ASSERT aborts the process in debug builds");
#else
    // The thread guard is the only thing between the GUI thread and a
    // multi-second blocking socket call (IMailService.h contract). This test
    // runs on the main thread (QTEST_GUILESS_MAIN), so the guard must fire.
    QTest::ignoreMessage(
        QtWarningMsg,
        QRegularExpression("blocking mail I/O invoked on the GUI thread"));
    mail::enforceBackgroundThread("TestMailCredentialLifecycle");
#endif
}

void TestMailCredentialLifecycle::noCredentialsWithoutLogin()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    UserManager::instance()->loadConfig(dir.path() + "/users.ini");
    UserManager::instance()->logout();

    MailService svc(UserManager::instance());
    QVERIFY(!svc.hasCredentials());
    QVERIFY(svc.senderEmail().isEmpty());
}

void TestMailCredentialLifecycle::credentialsCachedForCurrentUser()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    UserManager::instance()->loadConfig(dir.path() + "/users.ini");

    QString err;
    QVERIFY(UserManager::instance()->login("admin", "1", err));

    MailService svc(UserManager::instance());
    QVERIFY(!svc.hasCredentials());

    svc.setCredentials("bot@example.com", "s3cret", "Bot");
    QVERIFY(svc.hasCredentials());
    QCOMPARE(svc.senderEmail(), QString("bot@example.com"));

    // A fresh instance must see the same account — the cache is derived from
    // persisted user state, not from in-memory leftovers.
    MailService fresh(UserManager::instance());
    QVERIFY(fresh.hasCredentials());
    QCOMPARE(fresh.senderEmail(), QString("bot@example.com"));

    UserManager::instance()->logout();
}

void TestMailCredentialLifecycle::credentialsInvalidatedOnLogout()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    UserManager::instance()->loadConfig(dir.path() + "/users.ini");

    QString err;
    QVERIFY(UserManager::instance()->login("admin", "1", err));

    MailService svc(UserManager::instance());
    svc.setCredentials("bot@example.com", "s3cret", "Bot");
    QVERIFY(svc.hasCredentials());

    UserManager::instance()->logout();

    // P0 regression: stale credentials must NOT survive a logout.
    QVERIFY(!svc.hasCredentials());
    QVERIFY(svc.senderEmail().isEmpty());
}

QTEST_GUILESS_MAIN(TestMailCredentialLifecycle)
#include "TestMailCredentialLifecycle.moc"
