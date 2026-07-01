#ifndef CHAT_MESSAGE_RENDERER_H
#define CHAT_MESSAGE_RENDERER_H

#include <QString>
#include <QJsonArray>
#include <QList>
#include <QJsonObject>
#include "IAppContext.h"

class QTextBrowser;

class ChatMessageRenderer {
public:
    static void renderChatHistory(QTextBrowser* browser, IAppContext* ctx, const QList<QJsonObject>& history, bool isThinking);
    static QString buildMessageHtml(const QString &role, const QString &content, const QJsonArray &attachments, const QString &timestamp, int index);
};

#endif // CHAT_MESSAGE_RENDERER_H
