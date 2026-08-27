#include "ChatSessionStore.h"

#include "AppConfig.h"
#include "UserManager.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QUuid>

namespace ChatSessionStore {

void save(const QList<ChatSession> &sessions)
{
    QJsonArray sessionArray;
    for (const auto &session : sessions) {
        QJsonObject sessionObject;
        sessionObject["id"] = session.id;
        sessionObject["title"] = session.title;
        sessionObject["createdAt"] = session.createdAt.toString(Qt::ISODate);

        QJsonArray messages;
        for (const auto &message : session.messages) {
            messages.append(message);
        }
        sessionObject["messages"] = messages;
        sessionArray.append(sessionObject);
    }

    QFile file(sessionsPath());
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(sessionArray).toJson());
        file.close();
    }
}

QList<ChatSession> load()
{
    QList<ChatSession> sessions;
    QFile file(sessionsPath());
    if (!file.open(QIODevice::ReadOnly)) return sessions;

    const QJsonArray sessionArray = QJsonDocument::fromJson(file.readAll()).array();
    file.close();

    for (const auto &value : sessionArray) {
        const QJsonObject sessionObject = value.toObject();
        ChatSession session;
        session.id = sessionObject["id"].toString();
        session.title = sessionObject["title"].toString();
        session.createdAt = QDateTime::fromString(sessionObject["createdAt"].toString(), Qt::ISODate);
        for (const auto &message : sessionObject["messages"].toArray()) {
            session.messages.append(message.toObject());
        }
        sessions.append(session);
    }

    return sessions;
}

QString generateSessionId()
{
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

QString sessionsPath()
{
    QString currentUser = UserManager::instance()->currentUsername();
    if (currentUser.isEmpty()) currentUser = "default";

    const QString dir = QFileInfo(AppConfig::instance().configPath()).absolutePath() + "/";
    QDir().mkpath(dir);
    return dir + "chat_sessions_" + currentUser + ".json";
}

}
