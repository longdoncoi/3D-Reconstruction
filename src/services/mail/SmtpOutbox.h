#ifndef SMTPOUTBOX_H
#define SMTPOUTBOX_H

#include "IOutgoingMail.h"
#include "Global.h"

class ISmtpCredentialService;

/**
 * @brief IOutgoingMail adapter over SmtpMailer, backed by the credentials
 *        stored behind ISmtpCredentialService (composition-root injected).
 *
 * Lives in services/mail so that core code never links against the SMTP
 * transport directly.
 */
class APP_EXPORT SmtpOutbox final : public IOutgoingMail {
public:
    explicit SmtpOutbox(ISmtpCredentialService *credentials);

    bool send(const QString &from,
              const QString &to,
              const QString &subject,
              const QString &htmlBody,
              QString &errorMsg) override;

private:
    ISmtpCredentialService *m_credentials; ///< not owned; must outlive this adapter
};

#endif // SMTPOUTBOX_H
