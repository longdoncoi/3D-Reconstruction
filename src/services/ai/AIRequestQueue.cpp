#include "AIRequestQueue.h"

#include <QNetworkReply>

void AIRequestQueue::enqueue(const AICompletionRequest &request)
{
    m_queued.append(request);
}

void AIRequestQueue::prepend(const AICompletionRequest &request)
{
    m_queued.prepend(request);
}

bool AIRequestQueue::takeNext(AICompletionRequest &request)
{
    if (!m_pending.isEmpty() || m_queued.isEmpty()) return false;
    request = m_queued.takeFirst();
    return true;
}

void AIRequestQueue::registerReply(QNetworkReply *reply, const AICompletionRequest &request)
{
    m_pending.insert(reply, request);
}

bool AIRequestQueue::hasReply(QNetworkReply *reply) const
{
    return m_pending.contains(reply);
}

AICompletionRequest AIRequestQueue::takeReply(QNetworkReply *reply)
{
    return m_pending.take(reply);
}

QList<QNetworkReply *> AIRequestQueue::pendingReplies() const
{
    return m_pending.keys();
}

QList<AICompletionRequest> AIRequestQueue::takeAllPending()
{
    const QList<AICompletionRequest> requests = m_pending.values();
    m_pending.clear();
    return requests;
}

bool AIRequestQueue::hasWork() const
{
    return !m_pending.isEmpty() || !m_queued.isEmpty();
}

bool AIRequestQueue::hasPending() const
{
    return !m_pending.isEmpty();
}

bool AIRequestQueue::hasQueued() const
{
    return !m_queued.isEmpty();
}

bool AIRequestQueue::hasRequestForSession(const QString &sessionId) const
{
    for (auto it = m_pending.begin(); it != m_pending.end(); ++it) {
        if (it.value().sessionId == sessionId) return true;
    }
    for (const AICompletionRequest &request : m_queued) {
        if (request.sessionId == sessionId) return true;
    }
    return false;
}

int AIRequestQueue::insertIndexForSession(const QString &sessionId) const
{
    for (auto it = m_pending.begin(); it != m_pending.end(); ++it) {
        if (it.value().sessionId == sessionId) return it.value().insertAfterIndex;
    }
    for (const AICompletionRequest &request : m_queued) {
        if (request.sessionId == sessionId) return request.insertAfterIndex;
    }
    return -1;
}

void AIRequestQueue::removeQueuedForSession(const QString &sessionId)
{
    for (int i = m_queued.size() - 1; i >= 0; --i) {
        if (m_queued[i].sessionId == sessionId) m_queued.removeAt(i);
    }
}

QList<AICompletionRequest> AIRequestQueue::takeAllQueued()
{
    const QList<AICompletionRequest> requests = m_queued;
    m_queued.clear();
    return requests;
}

void AIRequestQueue::clear()
{
    m_pending.clear();
    m_queued.clear();
}
