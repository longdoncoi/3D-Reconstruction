#ifndef MAIL_SERVICE_H
#define MAIL_SERVICE_H

#include "IMailService.h"
#include "Global.h"

#include <QString>
#include <shared_mutex>

class IAuthService;
class IUserPreferences;
class IUserProfileService;

class APP_EXPORT MailService : public IMailService {
public:
    /// Three narrow contracts instead of the aggregate IUserManager: the mail
    /// stack needs "who am I", "my profile" and "my preferences" — it must not
    /// be able to reach the admin surface just because it happens to hold the
    /// same object.
    MailService(IAuthService* auth, IUserProfileService* profile,
                IUserPreferences* prefs);

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
    IAuthService* m_auth;            ///< injected by AppShell (composition root)
    IUserProfileService* m_profile;  ///< injected, same instance
    IUserPreferences* m_prefs;       ///< injected, same instance
    /// Refreshes the cached account state from the injected user services.
    /// Declared const because the cache is a pure derived value of user state
    /// — const getters call it without const_cast (members below are mutable).
    void loadFromCurrentUser() const;
    QString currentUsername() const;
    QString imapHost() const;
    bool openImap(QString &errorMsg, class QSslSocket &sock) const;
    bool sendImap(class QSslSocket &sock, const QString &tag, const QString &command, QString &response) const;

    mutable std::shared_mutex m_mutex;
    mutable QString m_email;
    mutable QString m_password;
    mutable QString m_displayName;
    mutable QString m_signature;
};

#endif // MAIL_SERVICE_H
