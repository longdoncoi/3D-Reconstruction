#include "MailMimeParser.h"

#include <QRegularExpression>

namespace {

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

    QString decoded = QString::fromUtf8(bytes);
    if (decoded.contains(QChar::ReplacementCharacter)) {
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

QString decodeMimeWords(const QString &value)
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

}

namespace MailMimeParser {

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

MailMessage parseFetchedMessage(const QString &uid, const QString &raw)
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

}
