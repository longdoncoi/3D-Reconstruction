#ifndef CHAT_MESSAGE_RENDERER_H
#define CHAT_MESSAGE_RENDERER_H

#include <QString>
#include <QJsonArray>
#include <QList>
#include <QJsonObject>
#include <QSet>
#include "IAppContext.h"

class QTextBrowser;

class ChatMessageRenderer {
public:
    static void renderChatHistory(QTextBrowser* browser, IAppContext* ctx, const QList<QJsonObject>& history, bool isThinking, int thinkingInsertIndex = -1);
    static QString buildMessageHtml(const QString &role, const QString &content, const QJsonArray &attachments,
                                    const QString &timestamp, int index, IAppContext* ctx,
                                    const QSet<QString>& resolvedActions);
};

#endif // CHAT_MESSAGE_RENDERER_H
