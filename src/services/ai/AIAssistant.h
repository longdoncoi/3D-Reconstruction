#ifndef AIASSISTANT_H
#define AIASSISTANT_H

#include <QObject>
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
#include "ChatSessionManager.h"
#include "AIRequestQueue.h"
#include "AIServerProcessController.h"

class ILanguageService;
class IUserManager;
class IAppPaths;

class APP_EXPORT AIAssistant : public IAIAssistantService {
    Q_OBJECT
public:
    explicit AIAssistant(ILanguageService* lang, IUserManager* users,
                         IAppPaths* paths, QString pythonInterpreter,
                         int serverPort, QObject *parent = nullptr,
                         QNetworkAccessManager* networkManager = nullptr);
    ~AIAssistant();

    void startServer(int modelIndex) override;
    void stopServer() override;
    void sendMessage(const QString &text, const QStringList &attachments = QStringList()) override;
    void sendMessageToSession(const QString &sessionId, const QString &text, const QStringList &attachments = QStringList()) override;
    void retryMessage(const QString &sessionId, int msgIndex) override;
    void retryAgentTask(const QString &sessionId, int msgIndex) override;
    void editMessage(const QString &sessionId, int msgIndex, const QString &newText) override;
    void switchModel(int index) override;
    void restartModel() override;
    void restartRAG() override;
    void restartAgent() override;
    void restartServer() override;

    // Agent mode
    void executeAgentTask(const QString &sessionId, const QString &task,
                          const QStringList &attachments = QStringList()) override;
    void approveAgentAction(const QString &sessionId, const QString &actionId) override;
    void rejectAgentAction(const QString &sessionId, const QString &actionId) override;
    void reportUiActionResult(const QString &requestId, bool success,
                              const QVariantMap &result = QVariantMap()) override;

    // Session management
    void newChat() override;                              // Create new session (keep old)
    void loadSession(const QString &sessionId) override; // Switch to existing session
    void deleteSession(const QString &sessionId) override;
    void clearHistory() override;
    void reloadSessions() override;

    QList<ChatSession> getSessions() const override { return m_sessions.sessions(); }
    QString currentSessionId() const override { return m_sessions.currentId(); }
    QList<QJsonObject> getHistory() const override;
    bool isThinking() const override { return m_isThinking; }
    bool isSessionThinking(const QString &sessionId) const override;
    int sessionThinkingInsertIndex(const QString &sessionId) const override;
    bool isServerRunning() const override { return m_serverReadyEmitted; }

// Signals are declared in IAIAssistantService

protected:
    /// Transition to "backend ready" and drain queued work. Called by the
    /// QProcess stdout/stderr handlers; protected so the request state machine
    /// can be driven deterministically in tests without a real server process.
    void markServerReady();

private slots:
    void onServerFinished(int exitCode, bool crashed);
    void onReplyFinished(QNetworkReply* reply);

private:
    ILanguageService* m_lang = nullptr;  ///< injected, never null in practice
    IUserManager* m_users = nullptr;     ///< injected, never null in practice
    IAppPaths* m_paths = nullptr;        ///< injected by AppShell (composition root)
    QString m_pythonInterpreter;         ///< injected by AppShell (config, default "python")
    int m_serverPort = 8080;             ///< injected by AppShell (config, default 8080)
    using QueuedCompletionRequest = AICompletionRequest;

    AIServerProcessController m_server;
    /// Injected transport (test seam) or owned by this object when null.
    QNetworkAccessManager *networkManager;
    ChatSessionManager m_sessions; ///< session collection + current pointer
    AIRequestQueue m_requests;     ///< queued work + in-flight replies
    // Maps requestId → {sessionId, insertAfterIndex} so that the
    // ui-action ACK response is inserted at the correct position
    // instead of being appended to the end of the conversation.
    QMap<QString, QPair<QString, int>> m_pendingUiActionSessions;
    bool m_isThinking = false;
    bool m_serverReadyEmitted = false;
    int m_currentModelIndex = 0;
    int m_serverRecoveryAttempts = 0;

    ChatSession* currentSession();
    ChatSession* getSession(const QString &sessionId);
    bool hasPendingWork() const;
    bool hasPendingRequestForSession(const QString &sessionId) const;
    QJsonObject buildCompletionPayload(const QList<QJsonObject> &messages) const;
    QJsonObject buildAgentPayload(const QList<QJsonObject> &messages, const QString &task,
                                  const QStringList &attachments) const;
    bool isRecoverableConnectionError(QNetworkReply::NetworkError error) const;
    void appendAssistantMessage(const QString &sessionId, const QString &content, int insertAfterIndex = -1);
    /// Clears all in-flight request state (queued requests, pending replies,
    /// pending UI-action sessions) and resets thinking/recovery flags.
    /// Called whenever the backend generation is torn down (stop/restart).
    void resetTransientRequestState();
    void enqueueCompletionRequest(const QString &sessionId, const QJsonObject &payload, int insertAfterIndex = -1);
    void processNextQueuedRequest();
    void saveAllSessions();
    void loadAllSessions();
};

#endif // AIASSISTANT_H
