#ifndef MAIL_THREAD_GUARD_H
#define MAIL_THREAD_GUARD_H

#include <QCoreApplication>
#include <QDebug>
#include <QThread>

namespace mail {

/**
 * @brief Enforces the synchronous-mail threading contract (IMailService.h).
 *
 * Every network entry point of the mail stack blocks for seconds (explicit
 * socket timeouts). Callers must invoke them from a background thread —
 * MailDockWidget / MailSettingsDialog / ForgotPasswordDialog use
 * blocking::run() (src/utils/BlockingCall.h), which always uses a dedicated
 * thread. A violation warns in every build and asserts in debug builds (same
 * fail-fast stance as ServiceRegistry).
 */
inline void enforceBackgroundThread(const char *who)
{
    const QCoreApplication *app = QCoreApplication::instance();
    if (app != nullptr && QThread::currentThread() == app->thread()) {
        qWarning() << who
                   << ": blocking mail I/O invoked on the GUI thread — run it via "
                      "QtConcurrent/QThread (see IMailService.h)";
        Q_ASSERT_X(false, "mail::enforceBackgroundThread",
                   "blocking mail I/O must not run on the GUI thread");
    }
}

} // namespace mail

#endif // MAIL_THREAD_GUARD_H
