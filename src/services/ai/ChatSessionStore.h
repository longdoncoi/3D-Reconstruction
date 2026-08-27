#ifndef CHAT_SESSION_STORE_H
#define CHAT_SESSION_STORE_H

#include "IAIAssistantService.h"

namespace ChatSessionStore {

void save(const QList<ChatSession> &sessions);
QList<ChatSession> load();
QString generateSessionId();
QString sessionsPath();

}

#endif // CHAT_SESSION_STORE_H
