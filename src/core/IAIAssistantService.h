#ifndef IAIASSISTANTSERVICE_H
#define IAIASSISTANTSERVICE_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QList>
#include <QJsonObject>
#include <QDateTime>
#include "Global.h"

struct ChatSession {
    QString id;           // UUID-like unique ID
    QString title;        // Auto-generated from first message
    QDateTime createdAt;
    QList<QJsonObject> messages;
};

class APP_EXPORT IAIAssistantService : public QObject {
    Q_OBJECT
public:
    virtual ~IAIAssistantService() = default;

    virtual void startServer(int modelIndex) = 0;
    virtual void stopServer() = 0;
    virtual void sendMessage(const QString &text, const QStringList &attachments = QStringList()) = 0;
    virtual void sendMessageToSession(const QString &sessionId, const QString &text, const QStringList &attachments = QStringList()) = 0;
    virtual void retryMessage(const QString &sessionId, int msgIndex) = 0;
    virtual void editMessage(const QString &sessionId, int msgIndex, const QString &newText) = 0;
    virtual void switchModel(int index) = 0;

    // Session management
    virtual void newChat() = 0;
    virtual void loadSession(const QString &sessionId) = 0;
    virtual void deleteSession(const QString &sessionId) = 0;
    virtual void clearHistory() = 0;
    virtual void reloadSessions() = 0;

    virtual QList<ChatSession> getSessions() const = 0;
    virtual QString currentSessionId() const = 0;
    virtual QList<QJsonObject> getHistory() const = 0;
    virtual bool isThinking() const = 0;
    virtual bool isSessionThinking(const QString &sessionId) const = 0;
    virtual bool isServerRunning() const = 0;

signals:
    void historyChanged();
    void sessionsChanged();
    void serverStatusChanged(const QString &status);
    void errorOccurred(const QString &error);
    void responseReceived();

protected:
    explicit IAIAssistantService(QObject* parent = nullptr) : QObject(parent) {}
};

#endif // IAIASSISTANTSERVICE_H
