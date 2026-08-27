#include "MailService.h"

#include "MailMimeParser.h"
#include "SmtpMailer.h"
#include "UserManager.h"

#include <QHostAddress>
#include <QHostInfo>
#include <QSslSocket>

#include <algorithm>

namespace {
constexpr int ImapTimeoutMs = 8000;
}

MailService::MailService()
{
    loadFromCurrentUser();
}

QString MailService::currentUsername() const
{
    auto *um = UserManager::instance();
    return um ? um->currentUsername() : QString();
}

void MailService::loadFromCurrentUser()
{
    auto *um = UserManager::instance();
    const QString username = currentUsername();
    if (!um || username.isEmpty()) return;

    std::unique_lock lock(m_mutex);
    m_email = um->getUserPref(username, "mail_email", um->currentUser().email);
    m_password = um->getUserPref(username, "mail_password");
    m_displayName = um->getUserPref(username, "mail_display_name", username);
    m_signature = um->getUserPref(username, "mail_signature",
                                  QString("<b>%1</b><br>3D-Reconstruction").arg(m_displayName));
}

void MailService::setCredentials(const QString& email,
                                 const QString& password,
                                 const QString& displayName)
{
    {
        std::unique_lock lock(m_mutex);
        m_email = email.trimmed();
        m_password = password;
        m_displayName = displayName.trimmed();
    }

    auto *um = UserManager::instance();
    const QString username = currentUsername();
    if (!um || username.isEmpty()) return;
    
    std::shared_lock lock(m_mutex);
    um->setUserPref(username, "mail_email", m_email);
    um->setUserPref(username, "mail_password", m_password);
    um->setUserPref(username, "mail_display_name", m_displayName);
}

QString MailService::signature() const
{
    std::shared_lock lock(m_mutex);
    return m_signature;
}

QString MailService::displayName() const
{
    std::shared_lock lock(m_mutex);
    return m_displayName;
}

void MailService::setSignature(const QString &signature)
{
    {
        std::unique_lock lock(m_mutex);
        m_signature = signature;
    }
    auto *um = UserManager::instance();
    const QString username = currentUsername();
    if (um && !username.isEmpty()) {
        um->setUserPref(username, "mail_signature", signature);
    }
}

bool MailService::hasCredentials() const
{
    const_cast<MailService*>(this)->loadFromCurrentUser();
    std::shared_lock lock(m_mutex);
    return !m_email.trimmed().isEmpty() && !m_password.isEmpty();
}

QString MailService::senderEmail() const
{
    const_cast<MailService*>(this)->loadFromCurrentUser();
    std::shared_lock lock(m_mutex);
    return m_email;
}

bool MailService::sendMail(const MailMessage& message, QString& errorMsg)
{
    loadFromCurrentUser();
    
    QString email, password, displayName, signature;
    {
        std::shared_lock lock(m_mutex);
        email = m_email;
        password = m_password;
        displayName = m_displayName;
        signature = m_signature;
    }

    if (email.trimmed().isEmpty() || password.isEmpty()) {
        errorMsg = "Mail account is not configured.";
        return false;
    }

    SmtpMailer mailer(email, password);
    return mailer.sendMail(email,
                           MailMimeParser::splitAddresses(message.to),
                           MailMimeParser::splitAddresses(message.cc),
                           MailMimeParser::splitAddresses(message.bcc),
                           message.subject,
                           MailMimeParser::htmlWithSignature(message.htmlBody, signature),
                           message.attachmentPaths,
                           displayName,
                           errorMsg);
}

QString MailService::imapHost() const
{
    std::shared_lock lock(m_mutex);
    const QString domain = m_email.section('@', 1).toLower();
    if (domain == "gmail.com") return "imap.gmail.com";
    if (domain == "hotmail.com" || domain == "outlook.com" || domain == "live.com") return "outlook.office365.com";
    if (domain == "yahoo.com") return "imap.mail.yahoo.com";
    return "imap." + domain;
}

bool MailService::openImap(QString &errorMsg, QSslSocket &sock) const
{
    // Check SSL availability first (OpenSSL DLLs must be present at runtime)
    if (!QSslSocket::supportsSsl()) {
        errorMsg = QString("TLS is not available. SSL library build: %1, runtime: %2. "
                           "Please ensure OpenSSL DLLs (libssl, libcrypto) are in the application directory or PATH.")
                       .arg(QSslSocket::sslLibraryBuildVersionString(),
                            QSslSocket::sslLibraryVersionString());
        return false;
    }

    sock.setPeerVerifyMode(QSslSocket::VerifyNone);

    // Resolve host and prefer IPv4
    const QString host = imapHost();
    const QHostInfo hostInfo = QHostInfo::fromName(host);
    QHostAddress ipv4Address;
    for (const QHostAddress &addr : hostInfo.addresses()) {
        if (addr.protocol() == QAbstractSocket::IPv4Protocol) {
            ipv4Address = addr;
            break;
        }
    }

    // Port 993 uses implicit SSL/TLS — must use connectToHostEncrypted()
    // (connectToHost + startClientEncryption is for STARTTLS on port 143)
    if (!ipv4Address.isNull()) {
        sock.connectToHostEncrypted(ipv4Address.toString(), 993, host);
    } else {
        sock.connectToHostEncrypted(host, 993);
    }

    if (!sock.waitForEncrypted(ImapTimeoutMs)) {
        errorMsg = "Cannot establish TLS with IMAP server: " + sock.errorString();
        return false;
    }

    if (!sock.waitForReadyRead(ImapTimeoutMs)) {
        errorMsg = "IMAP server did not send a greeting.";
        return false;
    }
    sock.readAll();
    return true;
}

bool MailService::sendImap(QSslSocket &sock, const QString &tag, const QString &command, QString &response) const
{
    response.clear();
    sock.write((tag + " " + command + "\r\n").toUtf8());
    sock.flush();
    while (sock.waitForReadyRead(ImapTimeoutMs)) {
        response += QString::fromUtf8(sock.readAll());
        if (response.contains("\r\n" + tag + " OK") || response.startsWith(tag + " OK") ||
            response.contains("\r\n" + tag + " NO") || response.contains("\r\n" + tag + " BAD")) {
            break;
        }
    }
    return response.contains(tag + " OK");
}

bool MailService::testConnection(QString& errorMsg)
{
    loadFromCurrentUser();
    
    QString email, password;
    {
        std::shared_lock lock(m_mutex);
        email = m_email;
        password = m_password;
    }

    if (email.trimmed().isEmpty() || password.isEmpty()) {
        errorMsg = "Mail account is not configured.";
        return false;
    }

    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return false;

    QString response;
    const QString login = QString("LOGIN \"%1\" \"%2\"").arg(email, password);
    if (!sendImap(sock, "A001", login, response)) {
        errorMsg = "IMAP login failed: " + response;
        return false;
    }
    sendImap(sock, "A002", "LOGOUT", response);
    return true;
}

QList<MailMessage> MailService::fetchInbox(int limit, QString& errorMsg)
{
    loadFromCurrentUser();
    QList<MailMessage> messages;
    
    QString email, password;
    {
        std::shared_lock lock(m_mutex);
        email = m_email;
        password = m_password;
    }

    if (email.trimmed().isEmpty() || password.isEmpty()) {
        errorMsg = "Mail account is not configured.";
        return messages;
    }

    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return messages;

    QString response;
    if (!sendImap(sock, "A001", QString("LOGIN \"%1\" \"%2\"").arg(email, password), response)) {
        errorMsg = "IMAP login failed: " + response;
        return messages;
    }
    if (!sendImap(sock, "A002", "SELECT INBOX", response)) {
        errorMsg = "Cannot open inbox: " + response;
        return messages;
    }
    if (!sendImap(sock, "A003", "UID SEARCH ALL", response)) {
        errorMsg = "Cannot search inbox: " + response;
        return messages;
    }

    QStringList uids;
    const QRegularExpression searchRe("\\* SEARCH ([^\\r\\n]+)");
    const auto match = searchRe.match(response);
    if (match.hasMatch()) {
        uids = match.captured(1).split(' ', Qt::SkipEmptyParts);
    }
    while (uids.size() > limit) uids.removeFirst();
    std::reverse(uids.begin(), uids.end());

    int tagIndex = 4;
    for (const QString &uid : uids) {
        const QString tag = QString("A%1").arg(tagIndex++, 3, 10, QLatin1Char('0'));
        const QString command = QString("UID FETCH %1 (FLAGS BODY.PEEK[]<0.500000>)").arg(uid);
        if (!sendImap(sock, tag, command, response)) {
            errorMsg = "IMAP fetch failed: " + response;
            break;
        }

        MailMessage parsed = MailMimeParser::parseFetchedMessage(uid, response);
        if (!parsed.uid.isEmpty()) {
            messages << parsed;
        }
    }

    QString logoutResponse;
    sendImap(sock, QString("A%1").arg(tagIndex++, 3, 10, QLatin1Char('0')), "LOGOUT", logoutResponse);
    return messages;
}

bool MailService::markRead(const QString& uid, QString& errorMsg)
{
    loadFromCurrentUser();
    
    QString email, password;
    {
        std::shared_lock lock(m_mutex);
        email = m_email;
        password = m_password;
    }
    
    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return false;
    QString response;
    if (!sendImap(sock, "A001", QString("LOGIN \"%1\" \"%2\"").arg(email, password), response)) {
        errorMsg = "IMAP login failed: " + response;
        return false;
    }
    sendImap(sock, "A002", "SELECT INBOX", response);
    const bool ok = sendImap(sock, "A003", QString("UID STORE %1 +FLAGS (\\Seen)").arg(uid), response);
    if (!ok) errorMsg = "Cannot mark message as read: " + response;
    sendImap(sock, "A004", "LOGOUT", response);
    return ok;
}

bool MailService::deleteMail(const QString& uid, QString& errorMsg)
{
    loadFromCurrentUser();
    
    QString email, password;
    {
        std::shared_lock lock(m_mutex);
        email = m_email;
        password = m_password;
    }

    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return false;
    QString response;
    if (!sendImap(sock, "A001", QString("LOGIN \"%1\" \"%2\"").arg(email, password), response)) {
        errorMsg = "IMAP login failed: " + response;
        return false;
    }
    sendImap(sock, "A002", "SELECT INBOX", response);
    bool ok = sendImap(sock, "A003", QString("UID STORE %1 +FLAGS (\\Deleted)").arg(uid), response);
    ok = ok && sendImap(sock, "A004", "EXPUNGE", response);
    if (!ok) errorMsg = "Cannot delete message: " + response;
    sendImap(sock, "A005", "LOGOUT", response);
    return ok;
}
