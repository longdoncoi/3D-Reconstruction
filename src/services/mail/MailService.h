#ifndef MAIL_SERVICE_H
#define MAIL_SERVICE_H

#include "IMailService.h"
#include "Global.h"

#include <QString>
#include <shared_mutex>

class APP_EXPORT MailService : public IMailService {
public:
    MailService();

    bool sendMail(const MailMessage& message, QString& errorMsg) override;
    QList<MailMessage> fetchInbox(int limit, QString& errorMsg) override;
    bool markRead(const QString& uid, QString& errorMsg) override;
    bool deleteMail(const QString& uid, QString& errorMsg) override;
    void setCredentials(const QString& email,
                        const QString& password,
                        const QString& displayName) override;
    bool testConnection(QString& errorMsg) override;
    bool hasCredentials() const override;
    QString senderEmail() const override;

    QString displayName() const;
    QString signature() const;
    void setSignature(const QString &signature);

private:
    void loadFromCurrentUser();
    QString currentUsername() const;
    QString imapHost() const;
    bool openImap(QString &errorMsg, class QSslSocket &sock) const;
    bool sendImap(class QSslSocket &sock, const QString &tag, const QString &command, QString &response) const;
    QString decodeMimeWords(const QString &value) const;
    MailMessage parseFetchedMessage(const QString &uid, const QString &raw) const;

    mutable std::shared_mutex m_mutex;
    QString m_email;
    QString m_password;
    QString m_displayName;
    QString m_signature;
};

#endif // MAIL_SERVICE_H
