#include "MailMessageFormatter.h"

#include <QRegularExpression>
#include <QTextDocument>

namespace {

QString decodeQuotedPrintable(const QString &input)
{
    QByteArray bytes;
    for (int i = 0; i < input.length(); ++i) {
        if (input[i] == '=' && i + 1 < input.length()) {
            if ((input[i + 1] == '\r' && i + 2 < input.length() && input[i + 2] == '\n') ||
                input[i + 1] == '\n') {
                i += (input[i + 1] == '\r') ? 2 : 1;
                continue;
            }

            if (i + 2 < input.length()) {
                const QString hexStr = input.mid(i + 1, 2);
                bool ok = false;
                const int byte = hexStr.toInt(&ok, 16);
                if (ok && byte >= 0 && byte <= 255) {
                    bytes.append(static_cast<char>(byte));
                    i += 2;
                    continue;
                }
            }
        }
        bytes.append(input[i].toLatin1());
    }
    return QString::fromUtf8(bytes);
}

bool looksLikeHtml(const QString &body)
{
    static const QRegularExpression htmlTagRe(
        "<\\s*/?\\s*(html|body|div|p|br|pre|span|table|tr|td|th|blockquote|b|strong|i|em|a)\\b",
        QRegularExpression::CaseInsensitiveOption);
    return htmlTagRe.match(body).hasMatch();
}

QString normalizeMailText(QString body)
{
    body = decodeQuotedPrintable(body);

    if (looksLikeHtml(body)) {
        QTextDocument doc;
        doc.setHtml(body);
        body = doc.toPlainText();
    }

    body.replace("\r\n", "\n");
    body.replace('\r', '\n');

    body.replace(QRegularExpression("\\s+(Vào\\s+[^\\n]{0,180}(?:đã\\s+)?viết\\s*:)",
                                    QRegularExpression::CaseInsensitiveOption),
                 "\n\n\\1\n");
    body.replace(QRegularExpression("\\s+(On\\s+[^\\n]{0,180}wrote\\s*:)",
                                    QRegularExpression::CaseInsensitiveOption),
                 "\n\n\\1\n");
    body.replace(QRegularExpression("\\s+((?:>\\s*)+)"), "\n\\1");

    return body;
}

QString renderMailBodyHtml(const QString &body)
{
    QString html;
    bool inQuote = false;
    bool quoteAfterReplyHeader = false;

    auto closeQuote = [&]() {
        if (inQuote) {
            html += "</div>";
            inQuote = false;
        }
    };

    for (QString line : body.split('\n')) {
        line = line.trimmed();
        if (line.isEmpty()) {
            closeQuote();
            continue;
        }

        int quoteDepth = 0;
        while (line.startsWith('>')) {
            ++quoteDepth;
            line = line.mid(1).trimmed();
        }
        if (line.isEmpty()) continue;

        const bool replyHeader =
            line.startsWith("Vào ", Qt::CaseInsensitive) ||
            line.startsWith("On ", Qt::CaseInsensitive);

        if (replyHeader) {
            closeQuote();
            html += QString("<div class='reply-header'>%1</div>").arg(line.toHtmlEscaped());
            quoteAfterReplyHeader = true;
            continue;
        }

        if (quoteAfterReplyHeader && quoteDepth == 0) {
            quoteDepth = 1;
        }

        if (quoteDepth > 0) {
            if (!inQuote) {
                html += "<div class='quoted-mail'>";
                inQuote = true;
            }
            const bool quoteMeta =
                line.startsWith("From:", Qt::CaseInsensitive) ||
                line.startsWith("To:", Qt::CaseInsensitive) ||
                line.startsWith("Date:", Qt::CaseInsensitive) ||
                line.startsWith("Subject:", Qt::CaseInsensitive);
            html += QString("<div class='%1'>%2</div>")
                        .arg(quoteMeta ? "quote-meta" : "quote-line",
                             line.toHtmlEscaped());
            continue;
        }

        closeQuote();
        html += QString("<p>%1</p>").arg(line.toHtmlEscaped());
    }

    closeQuote();
    return html.isEmpty() ? QString("<p></p>") : html;
}

QString previewStyle()
{
    return QString(
        "<style>"
        "body { font-family: 'Segoe UI', Arial, sans-serif; color: #e2e8f0; font-size: 14px; line-height: 1.7; "
        "       background-color: #0f172a; }"
        "h3 { font-size: 18px; font-weight: 600; margin: 0 0 14px 0; color: #f1f5f9; }"
        ".mail-header { background-color: #1e293b; padding: 18px 20px; border-radius: 10px; "
        "               margin-bottom: 20px; border: 1px solid #334155; }"
        ".mail-header-row { margin: 6px 0; font-size: 13px; }"
        ".mail-header-label { font-weight: 600; color: #94a3b8; display: inline-block; "
        "                     min-width: 55px; }"
        ".mail-header-value { color: #cbd5e1; }"
        ".mail-body { margin-top: 20px; word-wrap: break-word; color: #e2e8f0; }"
        ".mail-body p { margin: 8px 0; }"
        ".reply-header { margin: 20px 0 8px 0; padding: 8px 12px; color: #94a3b8; "
        "                font-weight: 600; font-size: 12px; background: #1e293b; "
        "                border-radius: 6px; }"
        ".quoted-mail { margin: 4px 0 0 0; padding: 10px 0 10px 14px; "
        "               border-left: 3px solid #3b82f6; color: #94a3b8; }"
        ".quote-line { margin: 3px 0; }"
        ".quote-meta { margin: 3px 0; color: #60a5fa; font-weight: 600; }"
        "a { color: #60a5fa; text-decoration: none; }"
        "a:hover { text-decoration: underline; color: #93c5fd; }"
        "table { border-collapse: collapse; width: 100%%; margin: 12px 0; }"
        "td, th { padding: 8px 12px; border: 1px solid #334155; color: #e2e8f0; }"
        "th { background-color: #1e293b; color: #f1f5f9; }"
        "pre { background-color: #1e293b; color: #e2e8f0; padding: 14px; "
        "      border-radius: 6px; overflow-x: auto; border: 1px solid #334155; }"
        "code { background-color: #1e293b; padding: 2px 6px; border-radius: 3px; "
        "       font-size: 13px; color: #f472b6; }"
        "img { max-width: 100%%; height: auto; border-radius: 6px; }"
        "hr { border: none; border-top: 1px solid #334155; margin: 16px 0; }"
        "</style>"
    );
}

QString headerHtml(const MailMessage &message)
{
    return QString(
        "<div class='mail-header'>"
        "<h3>%1</h3>"
        "<div class='mail-header-row'>"
        "  <span class='mail-header-label'>From:</span> "
        "  <span class='mail-header-value'>%2</span>"
        "</div>"
        "<div class='mail-header-row'>"
        "  <span class='mail-header-label'>To:</span> "
        "  <span class='mail-header-value'>%3</span>"
        "</div>"
        "<div class='mail-header-row'>"
        "  <span class='mail-header-label'>Date:</span> "
        "  <span class='mail-header-value'>%4</span>"
        "</div>"
        "</div>"
    ).arg(message.subject.toHtmlEscaped(),
          message.from.toHtmlEscaped(),
          message.to.toHtmlEscaped(),
          message.date.isValid() ? message.date.toString("ddd, dd MMM yyyy  hh:mm") : QString());
}

}

namespace MailMessageFormatter {

QString senderName(const QString &from)
{
    const QString trimmed = from.trimmed();
    const int angleBracket = trimmed.indexOf('<');
    if (angleBracket > 0) {
        QString name = trimmed.left(angleBracket).trimmed();
        if (name.startsWith('"') && name.endsWith('"')) {
            name = name.mid(1, name.length() - 2).trimmed();
        }
        if (!name.isEmpty()) return name;
    }

    static const QRegularExpression emailRe("<([^>]+)>");
    const auto match = emailRe.match(trimmed);
    return match.hasMatch() ? match.captured(1) : trimmed;
}

QString previewHtml(const MailMessage &message, const QString &emptyBodyText)
{
    QString bodyHtml = message.htmlBody.trimmed();
    if (bodyHtml.isEmpty()) {
        bodyHtml = QString("<p style='color:#64748b;font-style:italic;'>(%1)</p>")
                       .arg(emptyBodyText.toHtmlEscaped());
    } else if (!looksLikeHtml(bodyHtml)) {
        bodyHtml = renderMailBodyHtml(normalizeMailText(bodyHtml));
    }

    return previewStyle() + headerHtml(message) + "<div class='mail-body'>" + bodyHtml + "</div>";
}

}
