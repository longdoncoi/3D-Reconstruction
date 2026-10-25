#include "MailService.h"

#include "MailMimeParser.h"
#include "MailThreadGuard.h"
#include "SmtpMailer.h"
#include "IAuthService.h"
#include "IUserPreferences.h"
#include "IUserProfileService.h"

#include <QHostAddress>
#include <QHostInfo>
#include <QSslSocket>

#include <algorithm>

namespace {
constexpr int ImapTimeoutMs = 8000;

/// Quote a value for an IMAP quoted string (RFC 3501 §6.2.3).
///
/// Inside a quoted string the backslash and the double quote are escaped
/// with a backslash. Without this a password containing `"` or `\` ends the
/// string early and the rest of the password is parsed as protocol text —
/// the login fails with a confusing error, and a crafted password can inject
/// commands into the session.
QString imapQuoted(const QString &value)
{
    QString escaped = value;
    escaped.replace(QLatin1Char('\\'), QLatin1String("\\\\"));
    escaped.replace(QLatin1Char('"'), QLatin1String("\\\""));
    return escaped;
}

/// The LOGIN command with both values escaped. Every session opens with it,
/// so it is built in exactly one place.
QString imapLoginCommand(const QString &email, const QString &password)
{
    return QStringLiteral("LOGIN \"%1\" \"%2\"").arg(imapQuoted(email), imapQuoted(password));
}
} // namespace

MailService::MailService(IAuthService* auth, IUserProfileService* profile,
                         IUserPreferences* prefs)
    : m_auth(auth)
    , m_profile(profile)
    , m_prefs(prefs)
{
    loadFromCurrentUser();
}

QString MailService::currentUsername() const
{
    return m_auth ? m_auth->currentUsername() : QString();
}

void MailService::loadFromCurrentUser() const
{
    const QString username = currentUsername();

    if (!m_auth || !m_profile || !m_prefs || username.isEmpty()) {
        // Logout / not logged in: drop the cached credentials so a previous
        // session's account is never reused after the user changes state.
        std::unique_lock lock(m_mutex);
        m_email.clear();
        m_password.clear();
        m_displayName.clear();
        m_signature.clear();
        return;
    }

    // Read through the user services before taking our own lock — their
    // getters take the UserManager lock internally; keeping the two lock
    // domains non-overlapping avoids any lock-order coupling.
    const QString email = m_prefs->getUserPref(username, "mail_email", m_profile->currentUser().email);
    const QString password = m_prefs->getUserSecret(username, "mail_password");
    const QString displayName = m_prefs->getUserPref(username, "mail_display_name", username);
    const QString signature = m_prefs->getUserPref(
        username, "mail_signature",
        QString("<b>%1</b><br>3D-Reconstruction").arg(displayName));

    std::unique_lock lock(m_mutex);
    m_email = email;
    m_password = password;
    m_displayName = displayName;
    m_signature = signature;
}

void MailService::setCredentials(const QString& email,
                                 const QString& password,
                                 const QString& displayName)
{
    const QString trimmedEmail = email.trimmed();
    const QString trimmedName = displayName.trimmed();
    {
        std::unique_lock lock(m_mutex);
        m_email = trimmedEmail;
        m_password = password;
        m_displayName = trimmedName;
    }

    // Persist outside the lock — calling into the user services while holding
    // m_mutex would invert the natural lock order and is unnecessary here.
    const QString username = currentUsername();
    if (!m_prefs || username.isEmpty()) return;

    m_prefs->setUserPref(username, "mail_email", trimmedEmail);
    m_prefs->setUserSecret(username, "mail_password", password);
    m_prefs->setUserPref(username, "mail_display_name", trimmedName);
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
    const QString username = currentUsername();
    if (m_prefs && !username.isEmpty()) {
        m_prefs->setUserPref(username, "mail_signature", signature);
    }
}

bool MailService::hasCredentials() const
{
    loadFromCurrentUser();
    std::shared_lock lock(m_mutex);
    return !m_email.trimmed().isEmpty() && !m_password.isEmpty();
}

QString MailService::senderEmail() const
{
    loadFromCurrentUser();
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
    mail::enforceBackgroundThread("MailService::openImap");
    // Check SSL availability first (OpenSSL DLLs must be present at runtime)
    if (!QSslSocket::supportsSsl()) {
        errorMsg = QString("TLS is not available. SSL library build: %1, runtime: %2. "
                           "Please ensure OpenSSL DLLs (libssl, libcrypto) are in the application directory or PATH.")
                       .arg(QSslSocket::sslLibraryBuildVersionString(),
                            QSslSocket::sslLibraryVersionString());
        return false;
    }

    sock.setPeerVerifyMode(QSslSocket::VerifyPeer);

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

    // RECEIVING and SENDING are two different servers, usually with two
    // different credentials. Probing only IMAP reported "connected" for an
    // account that could read its inbox but could never send a message —
    // the one thing the button exists to find out.
    QString imapError;
    bool imapOk = false;
    {
        QSslSocket sock;
        if (openImap(imapError, sock)) {
            QString response;
            if (sendImap(sock, "A001", imapLoginCommand(email, password), response)) {
                imapOk = true;
                sendImap(sock, "A002", "LOGOUT", response);
            } else {
                imapError = "IMAP login failed: " + response;
            }
        }
    }

    QString smtpError;
    SmtpMailer mailer(email, password);
    const bool smtpOk = mailer.testConnection(smtpError);

    if (imapOk && smtpOk) return true;

    // Report the half that failed: "connection failed" with no direction
    // leaves the user guessing which server rejected them.
    if (!imapOk && !smtpOk)
        errorMsg = QStringLiteral("IMAP: %1\nSMTP: %2").arg(imapError, smtpError);
    else if (!imapOk)
        errorMsg = QStringLiteral("Receiving (IMAP) failed: %1").arg(imapError);
    else
        errorMsg = QStringLiteral("Sending (SMTP) failed: %1").arg(smtpError);
    return false;
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
    if (!sendImap(sock, "A001", imapLoginCommand(email, password), response)) {
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
    // A negative limit is not "unlimited": the trim loop would remove one
    // element per iteration from a list it has already emptied, which is
    // undefined behaviour. 0 is the honest reading of "no messages".
    limit = std::max(0, limit);
    if (limit < uids.size())
        uids = uids.mid(uids.size() - limit);
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
    if (!sendImap(sock, "A001", imapLoginCommand(email, password), response)) {
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
    if (!sendImap(sock, "A001", imapLoginCommand(email, password), response)) {
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
