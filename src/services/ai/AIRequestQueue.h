#ifndef AI_REQUEST_QUEUE_H
#define AI_REQUEST_QUEUE_H

#include <QJsonObject>
#include <QList>
#include <QMap>
#include <QString>

#include "Global.h"

class QNetworkReply;

/// One queued completion/agent request and its per-session insert position.
struct AICompletionRequest {
    QString sessionId;
    QJsonObject payload;
    int retryCount = 0;
    bool isAgent = false;
    bool isApproval = false;
    bool isUiActionAck = false;
    int insertAfterIndex = -1;
};

/**
 * @brief In-flight request bookkeeping for AIAssistant.
 *
 * The queue owns the pending-reply map and the FIFO of requests waiting for
 * the backend to become ready. It deliberately knows nothing about HTTP,
 * QProcess or UI signals: AIAssistant decides when to take the next request
 * and when a reply is registered/completed, which keeps the orchestrator from
 * growing another state machine.
 */
class APP_EXPORT AIRequestQueue {
public:
    void enqueue(const AICompletionRequest &request);
    void prepend(const AICompletionRequest &request);
    bool takeNext(AICompletionRequest &request);

    void registerReply(QNetworkReply *reply, const AICompletionRequest &request);
    bool hasReply(QNetworkReply *reply) const;
    AICompletionRequest takeReply(QNetworkReply *reply);
    QList<QNetworkReply *> pendingReplies() const;
    QList<AICompletionRequest> takeAllPending();

    bool hasWork() const;
    bool hasPending() const;
    bool hasQueued() const;
    bool hasRequestForSession(const QString &sessionId) const;
    int insertIndexForSession(const QString &sessionId) const;

    void removeQueuedForSession(const QString &sessionId);
    QList<AICompletionRequest> takeAllQueued();
    void clear();

private:
    QMap<QNetworkReply *, AICompletionRequest> m_pending;
    QList<AICompletionRequest> m_queued;
};

#endif // AI_REQUEST_QUEUE_H
