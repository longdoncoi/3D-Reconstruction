#include "ChatMessageRenderer.h"
#include "ChatTemplates.h"
#include "../../utils/HtmlUtilities.h"
#include <QTextBrowser>
#include <QUrl>
#include <QFileInfo>

void ChatMessageRenderer::renderChatHistory(QTextBrowser* browser, IAppContext* ctx, const QList<QJsonObject>& history, bool isThinking) {
    if (!browser) return;
    
    QString h = ChatTemplates::CSS;
    h += QString("<div class='chat-start'>%1</div>").arg(ctx->translate("ai.new_chat_start"));
    
    for (int i = 0; i < history.size(); ++i) {
        const auto &m = history[i];
        h += buildMessageHtml(m["role"].toString(), m["content"].toString(), m["attachments"].toArray(), m["timestamp"].toString(), i);
    }
    
    if (isThinking) {
        h += QString("<div class='typing'>%1</div>").arg(ctx->translate("ai.thinking"));
    }
    
    browser->setHtml(h);
    browser->moveCursor(QTextCursor::End);
}

QString ChatMessageRenderer::buildMessageHtml(const QString &role, const QString &content, const QJsonArray &attachments, const QString &timestamp, int index) {
    if (role == "assistant") {
        return ChatTemplates::AI_MESSAGE_CONTAINER.arg(HtmlUtilities::mdToHtml(content));
    } else {
        QString attHtml;
        if (!attachments.isEmpty()) {
            QString innerAtt;
            for (const auto &v : attachments) {
                QString path = v.toString();
                if (path.endsWith(".png") || path.endsWith(".jpg") || path.endsWith(".jpeg")) {
                    innerAtt += ChatTemplates::IMAGE_ATTACHMENT.arg(path).arg(QUrl::fromLocalFile(path).toString());
                } else {
                    innerAtt += ChatTemplates::FILE_ATTACHMENT
                                .arg(QUrl::fromLocalFile(path).toString())
                                .arg(QFileInfo(path).fileName());
                }
            }
            attHtml = ChatTemplates::ATTACHMENT_CONTAINER.arg(innerAtt);
        }

        QString textHtml;
        if (!content.isEmpty()) {
            QString escaped = content.toHtmlEscaped();
            escaped.replace("\n", "<br>");
            textHtml = ChatTemplates::USER_TEXT_TABLE.arg(escaped);
        }
        
        QString footerHtml = ChatTemplates::USER_ACTION_FOOTER;
        QDateTime dt = QDateTime::fromString(timestamp, "yyyy-MM-dd HH:mm:ss"); // Default chat format
        QString dateStr;
        if (dt.isValid()) {
            dateStr = dt.toString("MMM d");
        } else {
            dateStr = timestamp;
        }
        footerHtml = footerHtml.arg(dateStr).arg(index);

        return ChatTemplates::USER_MESSAGE_CONTAINER.arg(attHtml).arg(textHtml).arg(footerHtml);
    }
}
