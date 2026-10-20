#ifndef IMAIL_SERVICE_H
#define IMAIL_SERVICE_H

#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QList>
#include "Global.h"



/**
 * @brief Data structure representing a single email message.
 */
struct MailMessage {
    QString     uid;               ///< IMAP UID or local ID
    QString     from;              ///< Sender address
    QString     to;                ///< Recipient address(es), comma-separated
    QString     cc;                ///< CC addresses, comma-separated
    QString     bcc;               ///< BCC addresses, comma-separated (outgoing only)
    QString     subject;           ///< Subject line
    QString     htmlBody;          ///< HTML body content
    QStringList attachmentPaths;   ///< Absolute paths to attachment files
    QDateTime   date;              ///< Send/receive date
    bool        isRead = false;    ///< Read/unread status
};

/**
 * @brief Interface for the Mail Service.
 *
 * Provides contract for sending and receiving emails.
 *
 * Threading contract (enforced): every method below performs blocking network
 * I/O with multi-second timeouts. Callers MUST invoke them from a background
 * thread — MailDockWidget / MailSettingsDialog / ForgotPasswordDialog use
 * QtConcurrent::run. The implementation warns in every build and asserts in
 * debug builds when a blocking entry point is invoked on the GUI thread
 * (mail::enforceBackgroundThread).
 *
 * Usage:
 *   auto* mail = ctx->services()->get<IMailService>();
 *   mail->sendMail(msg, err);
 */
class IMailService {
public:
    /// ServiceRegistry lookup key (see ServiceRegistry.h).
    static constexpr const char *ServiceKey =
        "com.3dreconstruction.service.IMailService";

    virtual ~IMailService() = default;

    /**
     * @brief Send an email message.
     * @param message  Fully populated MailMessage (from, to, subject, htmlBody, attachments)
     * @param errorMsg Output: human-readable error description on failure
     * @return true on success
     */
    virtual bool sendMail(const MailMessage& message, QString& errorMsg) = 0;

    /**
     * @brief Fetch messages from the Inbox.
     * @param limit     Maximum number of messages to retrieve (most recent first)
     * @param errorMsg  Output: error description on failure
     * @return List of messages, empty on failure
     */
    virtual QList<MailMessage> fetchInbox(int limit, QString& errorMsg) = 0;

    /**
     * @brief Mark a message as read on the server.
     * @param uid      IMAP UID of the message
     * @param errorMsg Output: error description on failure
     * @return true on success
     */
    virtual bool markRead(const QString& uid, QString& errorMsg) = 0;

    /**
     * @brief Delete a message from the server.
     * @param uid      IMAP UID of the message
     * @param errorMsg Output: error description on failure
     * @return true on success
     */
    virtual bool deleteMail(const QString& uid, QString& errorMsg) = 0;

    /**
     * @brief Set/update SMTP+IMAP credentials for the current user.
     * @param email       Full email address (e.g. user@gmail.com)
     * @param password    App password or SMTP password
     * @param displayName Name shown in From header
     */
    virtual void setCredentials(const QString& email,
                                const QString& password,
                                const QString& displayName) = 0;

    /**
     * @brief Test SMTP + IMAP connectivity with current credentials.
     * @param errorMsg Output: error description on failure
     * @return true if both SMTP and IMAP connections succeed
     */
    virtual bool testConnection(QString& errorMsg) = 0;

    /**
     * @brief Returns whether credentials have been configured.
     */
    virtual bool hasCredentials() const = 0;

    /**
     * @brief Returns the configured sender email address.
     */
    virtual QString senderEmail() const = 0;
};

#endif // IMAIL_SERVICE_H
