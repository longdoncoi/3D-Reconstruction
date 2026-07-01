#ifndef AIASSISTANT_H
#define AIASSISTANT_H

#include <QObject>
#include <QProcess>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QList>
#include <QMap>
#include <QJsonObject>
#include <QJsonArray>
#include <QStringList>
#include <QDateTime>
#include "Global.h"
#include "IAIAssistantService.h"

class APP_EXPORT AIAssistant : public IAIAssistantService {
    Q_OBJECT
public:
    explicit AIAssistant(QObject *parent = nullptr);
    ~AIAssistant();

    void startServer(int modelIndex) override;
    void stopServer() override;
    void sendMessage(const QString &text, const QStringList &attachments = QStringList()) override;
    void sendMessageToSession(const QString &sessionId, const QString &text, const QStringList &attachments = QStringList()) override;
    void retryMessage(const QString &sessionId, int msgIndex) override;
    void editMessage(const QString &sessionId, int msgIndex, const QString &newText) override;
    void switchModel(int index) override;

    // Session management
    void newChat() override;                              // Create new session (keep old)
    void loadSession(const QString &sessionId) override; // Switch to existing session
    void deleteSession(const QString &sessionId) override;
    void clearHistory() override;
    void reloadSessions() override;

    QList<ChatSession> getSessions() const override { return m_sessions; }
    QString currentSessionId() const override { return m_currentSessionId; }
    QList<QJsonObject> getHistory() const override;
    bool isThinking() const override { return m_isThinking; }
    bool isSessionThinking(const QString &sessionId) const override;
    bool isServerRunning() const override { return aiServerProcess->state() != QProcess::NotRunning; }

// Signals are declared in IAIAssistantService

private slots:
    void onProcessReadyRead();
    void onProcessError();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onReplyFinished(QNetworkReply* reply);

private:
    QProcess *aiServerProcess;
    QNetworkAccessManager *networkManager;
    QList<ChatSession> m_sessions;
    QString m_currentSessionId;
    QMap<QNetworkReply*, QString> m_pendingRequests;  // Map reply to sessionId for tracking multiple requests
    bool m_isThinking = false;
    bool m_serverReadyEmitted = false;

    ChatSession* currentSession();
    ChatSession* getSession(const QString &sessionId);
    void saveAllSessions();
    void loadAllSessions();
    QString getSessionsPath();
    QString generateSessionId();
};

#endif // AIASSISTANT_H
