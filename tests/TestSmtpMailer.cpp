#include <QtTest>

#include <QFuture>
#include <QCoreApplication>
#include <QThread>
#include <QThreadPool>
#include <QCoreApplication>

#include "BlockingCall.h"
#include "SmtpMailer.h"

/**
 * @brief Contract of the SMTP client (src/services/mail/SmtpMailer.cpp).
 *
 * The full SMTP dialogue needs a TLS server, so this test pins the parts
 * that must behave without one: the input validation and the failure
 * reporting.  Everything runs off the GUI thread — the mail stack asserts in
 * debug builds when blocking I/O happens on the GUI thread
 * (mail::enforceBackgroundThread).
 */
class TestSmtpMailer : public QObject {
    Q_OBJECT

private slots:
    void rejectsAMissingSender();
    void rejectsAMissingRecipient();
    void theMailCallRunsOffTheGuiThread();
    void reportsAnUnreachableServer();
};

namespace {

/// Runs the blocking call on a worker thread and returns the result.
struct MailResult {
    bool ok = false;
    QString error;
    Qt::HANDLE callerThread = nullptr;
};

MailResult sendInBackground(SmtpMailer &mailer, const QString &from, const QString &to)
{
    MailResult result;
    // blocking::run() is what the application uses: it always uses a
    // dedicated thread, while QtConcurrent::run may execute the task inline.
    QFuture<MailResult> future = blocking::run([&mailer, &result, from, to] {
        result.callerThread = QThread::currentThreadId();
        result.ok = mailer.sendMail(from, to, QStringLiteral("subject"),
                                     QStringLiteral("<p>body</p>"), result.error);
        return result;
    });
    // The ctest timeout is the backstop: the failure paths return at once
    // (no DNS answer, no server), so the wait cannot hang in practice.
    future.waitForFinished();
    return future.result();
}

} // namespace

void TestSmtpMailer::rejectsAMissingSender()
{
    SmtpMailer mailer(QStringLiteral("user@example.com"), QStringLiteral("secret"));
    const MailResult result = sendInBackground(mailer, QString(), QStringLiteral("to@example.com"));
    QVERIFY(!result.ok);
    QVERIFY2(result.error.contains(QStringLiteral("Missing sender or recipient")),
             qPrintable(result.error));
}

void TestSmtpMailer::rejectsAMissingRecipient()
{
    SmtpMailer mailer(QStringLiteral("user@example.com"), QStringLiteral("secret"));
    const MailResult result = sendInBackground(mailer, QStringLiteral("from@example.com"), QString());
    QVERIFY(!result.ok);
    QVERIFY2(result.error.contains(QStringLiteral("Missing sender or recipient")),
             qPrintable(result.error));
}

void TestSmtpMailer::theMailCallRunsOffTheGuiThread()
{
    SmtpMailer mailer(QStringLiteral("user@example.com"), QStringLiteral("secret"));
    const MailResult result = sendInBackground(mailer, QStringLiteral("from@example.com"),
                                               QStringLiteral("to@example.com"));
    // The blocking call must never run on the GUI thread: blocking::run()
    // exists exactly because a pool may execute a task inline.
    QVERIFY2(result.callerThread != QThread::currentThreadId(),
             "the SMTP call ran on the GUI thread — the blocking helper was bypassed");
}

void TestSmtpMailer::reportsAnUnreachableServer()
{
    // The host is derived from the sender's domain: an .invalid domain can
    // never resolve, so the connection attempt fails immediately.
    SmtpMailer mailer(QStringLiteral("user@nonexistent.invalid"), QStringLiteral("secret"));
    const MailResult result = sendInBackground(mailer, QStringLiteral("from@nonexistent.invalid"),
                                               QStringLiteral("to@nonexistent.invalid"));
    QVERIFY(!result.ok);
    QVERIFY2(!result.error.isEmpty(), "a failed send must report an error");
    qInfo() << "SMTP failure reported as:" << result.error;
}

QTEST_GUILESS_MAIN(TestSmtpMailer)
#include "TestSmtpMailer.moc"
