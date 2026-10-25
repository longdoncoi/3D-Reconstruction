#include "SmtpOutbox.h"
#include "ISmtpCredentialService.h"
#include "SmtpMailer.h"

SmtpOutbox::SmtpOutbox(ISmtpCredentialService *credentials)
    : m_credentials(credentials)
{
}

bool SmtpOutbox::send(const QString &from,
                      const QString &to,
                      const QString &subject,
                      const QString &htmlBody,
                      QString &errorMsg)
{
    if (!m_credentials) {
        errorMsg = QStringLiteral("No user manager available for mail credentials.");
        return false;
    }

    const QString senderEmail = m_credentials->smtpSenderEmail();
    const QString senderPass  = m_credentials->smtpSenderPassword();
    if (senderEmail.trimmed().isEmpty()) {
        errorMsg = QStringLiteral("Chưa cấu hình email gửi. Vui lòng liên hệ Admin.");
        return false;
    }
    if (senderEmail.compare(from, Qt::CaseInsensitive) != 0) {
        errorMsg = QStringLiteral("Sender address does not match the configured SMTP account.");
        return false;
    }

    SmtpMailer mailer(senderEmail, senderPass);
    return mailer.sendMail(senderEmail, to, subject, htmlBody, errorMsg);
}
