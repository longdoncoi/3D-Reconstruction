#include "MailService.h"

#include "SmtpMailer.h"
#include "UserManager.h"

#include <QHostAddress>
#include <QHostInfo>
#include <QRegularExpression>
#include <QSslSocket>

#include <algorithm>

namespace {
constexpr int ImapTimeoutMs = 8000;

QStringList splitAddresses(const QString &value)
{
    QStringList result;
    const QStringList parts = value.split(QRegularExpression("[,;]"), Qt::SkipEmptyParts);
    for (QString part : parts) {
        part = part.trimmed();
        const QRegularExpression angle("<([^>]+)>");
        const auto match = angle.match(part);
        if (match.hasMatch()) part = match.captured(1).trimmed();
        if (!part.isEmpty()) result << part;
    }
    return result;
}

QString htmlWithSignature(QString html, const QString &signature)
{
    if (signature.trimmed().isEmpty()) return html;
    return html + "<br><br><div style=\"color:#64748b;\">" + signature + "</div>";
}

QString normalizeNewlines(QString text)
{
    text.replace("\r\n", "\n");
    text.replace('\r', '\n');
    return text;
}

void splitHeaderBody(const QString &raw, QString &headers, QString &body)
{
    const int crlfEnd = raw.indexOf("\r\n\r\n");
    if (crlfEnd >= 0) {
        headers = raw.left(crlfEnd);
        body = raw.mid(crlfEnd + 4);
        return;
    }

    const int lfEnd = raw.indexOf("\n\n");
    if (lfEnd >= 0) {
        headers = raw.left(lfEnd);
        body = raw.mid(lfEnd + 2);
        return;
    }

    headers.clear();
    body = raw;
}

QString headerValue(const QString &headers, const QString &name)
{
    const QString target = name.toLower();
    QString currentName;
    QString currentValue;

    auto flush = [&]() -> QString {
        if (currentName.toLower() == target) {
            return currentValue.simplified();
        }
        return {};
    };

    const QStringList lines = normalizeNewlines(headers).split('\n');
    for (const QString &line : lines) {
        if ((line.startsWith(' ') || line.startsWith('\t')) && !currentName.isEmpty()) {
            currentValue += " " + line.trimmed();
            continue;
        }

        const QString found = flush();
        if (!found.isEmpty()) return found;

        const int colon = line.indexOf(':');
        if (colon < 0) {
            currentName.clear();
            currentValue.clear();
            continue;
        }

        currentName = line.left(colon).trimmed();
        currentValue = line.mid(colon + 1).trimmed();
    }

    return flush();
}

QString contentTypeParameter(const QString &contentType, const QString &name)
{
    const QRegularExpression re(
        QString(R"re((?:^|;)\s*%1\s*=\s*(?:"([^"]*)"|([^;\s]+)))re")
            .arg(QRegularExpression::escape(name)),
        QRegularExpression::CaseInsensitiveOption);
    const auto match = re.match(contentType);
    if (!match.hasMatch()) return {};
    return !match.captured(1).isEmpty() ? match.captured(1).trimmed()
                                        : match.captured(2).trimmed();
}

QString mediaType(const QString &contentType)
{
    const QString type = contentType.section(';', 0, 0).trimmed().toLower();
    return type.isEmpty() ? QStringLiteral("text/plain") : type;
}

QByteArray decodeQuotedPrintableBytes(const QString &input, bool underscoreAsSpace = false)
{
    QByteArray bytes;
    bytes.reserve(input.size());

    for (int i = 0; i < input.length(); ++i) {
        const QChar ch = input[i];
        if (underscoreAsSpace && ch == '_') {
            bytes.append(' ');
            continue;
        }

        if (ch == '=' && i + 1 < input.length()) {
            if ((input[i + 1] == '\r' && i + 2 < input.length() && input[i + 2] == '\n') ||
                input[i + 1] == '\n') {
                i += (input[i + 1] == '\r') ? 2 : 1;
                continue;
            }

            if (i + 2 < input.length()) {
                bool ok = false;
                const int byte = input.mid(i + 1, 2).toInt(&ok, 16);
                if (ok && byte >= 0 && byte <= 255) {
                    bytes.append(static_cast<char>(byte));
                    i += 2;
                    continue;
                }
            }
        }

        bytes.append(ch.toLatin1());
    }

    return bytes;
}

QString decodeBytesForCharset(const QByteArray &bytes, QString charset)
{
    charset = charset.trimmed().remove('"').toLower();
    if (charset.isEmpty() || charset == "utf-8" || charset == "utf8" || charset == "us-ascii") {
        return QString::fromUtf8(bytes);
    }

    if (charset == "iso-8859-1" || charset == "latin1" || charset == "latin-1" ||
        charset == "windows-1252" || charset == "cp1252") {
        return QString::fromLatin1(bytes);
    }

    // Map Asian and Russian charsets to use QStringDecoder if needed,
    // For simplicity, we fallback to fromUtf8 and check for ReplacementCharacter
    QString decoded = QString::fromUtf8(bytes);
    if (decoded.contains(QChar::ReplacementCharacter)) {
        // Fallback to latin1 or system local
        decoded = QString::fromLocal8Bit(bytes);
        if (decoded.contains(QChar::ReplacementCharacter)) {
            decoded = QString::fromLatin1(bytes);
        }
    }
    return decoded;
}

QString decodeTransferBody(const QString &body, QString transferEncoding, QString charset)
{
    transferEncoding = transferEncoding.trimmed().toLower();
    charset = charset.trimmed().remove('"');

    if (transferEncoding.contains("base64")) {
        QString clean = body;
        clean.remove(QRegularExpression("\\s+"));
        return decodeBytesForCharset(QByteArray::fromBase64(clean.toLatin1()), charset);
    }

    const bool looksQuotedPrintable =
        QRegularExpression("=[0-9A-Fa-f]{2}|=\\r?\\n").match(body).hasMatch();
    if (transferEncoding.contains("quoted-printable") ||
        (transferEncoding.isEmpty() && looksQuotedPrintable)) {
        return decodeBytesForCharset(decodeQuotedPrintableBytes(body), charset);
    }

    return body;
}

struct DecodedBody {
    QString body;
    bool isHtml = false;
    int score = 0;
};

bool looksLikeHtml(const QString &body)
{
    static const QRegularExpression htmlTagRe(
        "<\\s*/?\\s*(html|body|div|p|br|pre|span|table|tr|td|th|blockquote|b|strong|i|em|a)\\b",
        QRegularExpression::CaseInsensitiveOption);
    return htmlTagRe.match(body).hasMatch();
}

QList<QString> splitMimeParts(const QString &body, const QString &boundary)
{
    QList<QString> parts;
    if (boundary.isEmpty()) return parts;

    const QString normalized = normalizeNewlines(body);
    const QString marker = "--" + boundary;
    int pos = normalized.indexOf(marker);

    while (pos >= 0) {
        const int markerEnd = normalized.indexOf('\n', pos);
        const QString markerLine = normalized.mid(pos, markerEnd < 0 ? -1 : markerEnd - pos).trimmed();
        if (markerLine == marker + "--") break;
        if (markerLine != marker) {
            pos = normalized.indexOf(marker, pos + marker.length());
            continue;
        }

        const int partStart = markerEnd < 0 ? normalized.length() : markerEnd + 1;
        int nextMarker = normalized.indexOf("\n" + marker, partStart);
        if (nextMarker < 0) nextMarker = normalized.length();

        const QString part = normalized.mid(partStart, nextMarker - partStart).trimmed();
        if (!part.isEmpty()) parts.append(part);

        pos = nextMarker < normalized.length() ? nextMarker + 1 : -1;
    }

    return parts;
}

DecodedBody extractBestBody(const QString &headers, const QString &body)
{
    const QString contentType = headerValue(headers, "Content-Type");
    const QString type = mediaType(contentType);
    const QString disposition = headerValue(headers, "Content-Disposition").toLower();
    if (disposition.startsWith("attachment")) return {};

    if (type.startsWith("multipart/")) {
        const QString boundary = contentTypeParameter(contentType, "boundary");
        DecodedBody best;
        for (const QString &partRaw : splitMimeParts(body, boundary)) {
            QString partHeaders;
            QString partBody;
            splitHeaderBody(partRaw, partHeaders, partBody);
            const DecodedBody part = extractBestBody(partHeaders, partBody);
            if (!part.body.trimmed().isEmpty() && part.score > best.score) {
                best = part;
            }
        }
        return best;
    }

    if (type == "message/rfc822") {
        QString nestedHeaders;
        QString nestedBody;
        splitHeaderBody(body, nestedHeaders, nestedBody);
        return extractBestBody(nestedHeaders, nestedBody);
    }

    if (type.startsWith("text/")) {
        const QString charset = contentTypeParameter(contentType, "charset");
        QString decoded = decodeTransferBody(body,
                                             headerValue(headers, "Content-Transfer-Encoding"),
                                             charset.isEmpty() ? QStringLiteral("utf-8") : charset);
        decoded = normalizeNewlines(decoded).trimmed();
        if (decoded.isEmpty()) return {};

        const bool isHtml = type == "text/html" || looksLikeHtml(decoded);
        return {decoded, isHtml, isHtml ? 100 : 50};
    }

    return {};
}

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
    m_email = email.trimmed();
    m_password = password;
    m_displayName = displayName.trimmed();

    auto *um = UserManager::instance();
    const QString username = currentUsername();
    if (!um || username.isEmpty()) return;
    um->setUserPref(username, "mail_email", m_email);
    um->setUserPref(username, "mail_password", m_password);
    um->setUserPref(username, "mail_display_name", m_displayName);
}

QString MailService::signature() const
{
    return m_signature;
}

QString MailService::displayName() const
{
    return m_displayName;
}

void MailService::setSignature(const QString &signature)
{
    m_signature = signature;
    auto *um = UserManager::instance();
    const QString username = currentUsername();
    if (um && !username.isEmpty()) {
        um->setUserPref(username, "mail_signature", signature);
    }
}

bool MailService::hasCredentials() const
{
    const_cast<MailService*>(this)->loadFromCurrentUser();
    return !m_email.trimmed().isEmpty() && !m_password.isEmpty();
}

QString MailService::senderEmail() const
{
    const_cast<MailService*>(this)->loadFromCurrentUser();
    return m_email;
}

bool MailService::sendMail(const MailMessage& message, QString& errorMsg)
{
    loadFromCurrentUser();
    if (!hasCredentials()) {
        errorMsg = "Mail account is not configured.";
        return false;
    }

    SmtpMailer mailer(m_email, m_password);
    return mailer.sendMail(m_email,
                           splitAddresses(message.to),
                           splitAddresses(message.cc),
                           splitAddresses(message.bcc),
                           message.subject,
                           htmlWithSignature(message.htmlBody, m_signature),
                           message.attachmentPaths,
                           m_displayName,
                           errorMsg);
}

QString MailService::imapHost() const
{
    const QString domain = m_email.section('@', 1).toLower();
    if (domain == "gmail.com") return "imap.gmail.com";
    if (domain == "hotmail.com" || domain == "outlook.com" || domain == "live.com") return "outlook.office365.com";
    if (domain == "yahoo.com") return "imap.mail.yahoo.com";
    return "imap." + domain;
}

bool MailService::openImap(QString &errorMsg, QSslSocket &sock) const
{
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

    // Connect using IPv4 if available, then upgrade to TLS
    if (!ipv4Address.isNull()) {
        sock.connectToHost(ipv4Address, 993);
    } else {
        sock.connectToHost(host, 993);
    }

    if (!sock.waitForConnected(ImapTimeoutMs)) {
        errorMsg = "Cannot connect to IMAP server: " + sock.errorString();
        return false;
    }

    sock.startClientEncryption();
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
    if (!hasCredentials()) {
        errorMsg = "Mail account is not configured.";
        return false;
    }

    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return false;

    QString response;
    const QString login = QString("LOGIN \"%1\" \"%2\"").arg(m_email, m_password);
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
    if (!hasCredentials()) {
        errorMsg = "Mail account is not configured.";
        return messages;
    }

    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return messages;

    QString response;
    if (!sendImap(sock, "A001", QString("LOGIN \"%1\" \"%2\"").arg(m_email, m_password), response)) {
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

        MailMessage parsed = parseFetchedMessage(uid, response);
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
    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return false;
    QString response;
    if (!sendImap(sock, "A001", QString("LOGIN \"%1\" \"%2\"").arg(m_email, m_password), response)) {
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
    QSslSocket sock;
    if (!openImap(errorMsg, sock)) return false;
    QString response;
    if (!sendImap(sock, "A001", QString("LOGIN \"%1\" \"%2\"").arg(m_email, m_password), response)) {
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

QString MailService::decodeMimeWords(const QString &value) const
{
    QString decoded;
    QString source = value;
    source.replace(QRegularExpression("\\r?\\n[\\t ]+"), " ");

    static const QRegularExpression encodedWordRe(
        "=\\?([^?\\s]+)\\?([bBqQ])\\?([^?]*)\\?=");

    int cursor = 0;
    bool previousWasEncodedWord = false;
    auto it = encodedWordRe.globalMatch(source);
    while (it.hasNext()) {
        const auto match = it.next();
        const QString gap = source.mid(cursor, match.capturedStart() - cursor);
        if (!(previousWasEncodedWord && gap.trimmed().isEmpty())) {
            decoded += gap;
        }

        const QString charset = match.captured(1);
        const QString encoding = match.captured(2).toLower();
        const QString payload = match.captured(3);

        QByteArray bytes;
        if (encoding == "b") {
            bytes = QByteArray::fromBase64(payload.toLatin1());
        } else {
            bytes = decodeQuotedPrintableBytes(payload, true);
        }

        decoded += decodeBytesForCharset(bytes, charset);
        cursor = match.capturedEnd();
        previousWasEncodedWord = true;
    }

    decoded += source.mid(cursor);
    return decoded.trimmed();
}

MailMessage MailService::parseFetchedMessage(const QString &uid, const QString &raw) const
{
    MailMessage msg;
    msg.uid = uid;
    msg.isRead = raw.contains("\\Seen");

    if (raw.isEmpty()) return msg;

    QString messageContent = raw;
    const QRegularExpression literalRe(
        R"((?:RFC822(?:\.PEEK)?|BODY\[\])(?:<\d+(?:\.\d+)?>)?\s*\{\d+\}\r?\n)",
        QRegularExpression::CaseInsensitiveOption);
    const auto literalMatch = literalRe.match(raw);
    if (literalMatch.hasMatch()) {
        const int contentStart = literalMatch.capturedEnd();
        int contentEnd = raw.lastIndexOf("\r\n)\r\n");
        if (contentEnd < 0) contentEnd = raw.lastIndexOf("\n)\n");
        if (contentEnd < 0) contentEnd = raw.lastIndexOf("\r\n)");
        if (contentEnd < 0) contentEnd = raw.lastIndexOf("\n)");
        if (contentEnd > contentStart) {
            messageContent = raw.mid(contentStart, contentEnd - contentStart);
        } else {
            messageContent = raw.mid(contentStart);
        }
    }

    if (messageContent.isEmpty()) return msg;

    QString headers;
    QString body;
    splitHeaderBody(messageContent, headers, body);

    msg.from = decodeMimeWords(headerValue(headers, "From"));
    msg.to = decodeMimeWords(headerValue(headers, "To"));
    msg.cc = decodeMimeWords(headerValue(headers, "Cc"));
    msg.subject = decodeMimeWords(headerValue(headers, "Subject"));
    msg.date = QDateTime::fromString(headerValue(headers, "Date"), Qt::RFC2822Date);

    DecodedBody decodedBody = extractBestBody(headers, body);
    if (decodedBody.body.trimmed().isEmpty()) {
        const QString contentType = headerValue(headers, "Content-Type");
        const QString charset = contentTypeParameter(contentType, "charset");
        decodedBody.body = decodeTransferBody(body,
                                              headerValue(headers, "Content-Transfer-Encoding"),
                                              charset.isEmpty() ? QStringLiteral("utf-8") : charset);
        decodedBody.body = normalizeNewlines(decodedBody.body).trimmed();
        decodedBody.isHtml = looksLikeHtml(decodedBody.body);
    }

    if (decodedBody.body.trimmed().isEmpty()) {
        msg.htmlBody.clear();
    } else if (decodedBody.isHtml) {
        msg.htmlBody = decodedBody.body.trimmed();
    } else {
        msg.htmlBody = QString("<pre style=\"white-space:pre-wrap; font-family:inherit;\">%1</pre>")
                           .arg(decodedBody.body.toHtmlEscaped());
    }
    return msg;
}
