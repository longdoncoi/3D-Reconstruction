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
    
    for (const auto &m : history) {
        h += buildMessageHtml(m["role"].toString(), m["content"].toString(), m["attachments"].toArray());
    }
    
    if (isThinking) {
        h += QString("<div class='typing'>%1</div>").arg(ctx->translate("ai.thinking"));
    }
    
    browser->setHtml(h);
    browser->moveCursor(QTextCursor::End);
}

QString ChatMessageRenderer::buildMessageHtml(const QString &role, const QString &content, const QJsonArray &attachments) {
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
        return ChatTemplates::USER_MESSAGE_CONTAINER.arg(attHtml).arg(textHtml);
    }
}
