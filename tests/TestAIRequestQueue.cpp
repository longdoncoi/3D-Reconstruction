#include <QtTest>
#include <QNetworkReply>

#include "AIRequestQueue.h"

class DummyReply : public QNetworkReply {
public:
    explicit DummyReply(QObject *parent = nullptr) : QNetworkReply(parent) {}
    void abort() override {}
protected:
    qint64 readData(char *, qint64) override { return -1; }
};

class TestAIRequestQueue : public QObject {
    Q_OBJECT

private slots:
    void fifoAndSessionQueries();
    void pendingRepliesAndRetryHandling();
    void sessionRemovalAndClear();
};

namespace {
AICompletionRequest makeRequest(const QString &sessionId, int insertAfterIndex = -1)
{
    AICompletionRequest request;
    request.sessionId = sessionId;
    request.payload["session"] = sessionId;
    request.insertAfterIndex = insertAfterIndex;
    return request;
}
} // namespace

void TestAIRequestQueue::fifoAndSessionQueries()
{
    AIRequestQueue queue;
    QVERIFY(!queue.hasWork());
    QVERIFY(!queue.hasQueued());
    QVERIFY(!queue.hasPending());

    const AICompletionRequest first = makeRequest("a", 2);
    const AICompletionRequest second = makeRequest("b");
    queue.enqueue(first);
    queue.enqueue(second);
    QVERIFY(queue.hasWork());
    QVERIFY(queue.hasQueued());
    QVERIFY(queue.hasRequestForSession("a"));
    QVERIFY(queue.hasRequestForSession("b"));
    QVERIFY(!queue.hasRequestForSession("c"));
    QCOMPARE(queue.insertIndexForSession("a"), 2);
    QCOMPARE(queue.insertIndexForSession("b"), -1);
    QCOMPARE(queue.insertIndexForSession("c"), -1);

    AICompletionRequest taken;
    QVERIFY(queue.takeNext(taken));
    QCOMPARE(taken.sessionId, QStringLiteral("a"));
    QCOMPARE(taken.payload.value("session").toString(), QStringLiteral("a"));
    QVERIFY(queue.hasQueued());
    QVERIFY(queue.takeNext(taken));
    QCOMPARE(taken.sessionId, QStringLiteral("b"));
    QVERIFY(!queue.hasWork());
    QVERIFY(!queue.takeNext(taken));
}

void TestAIRequestQueue::pendingRepliesAndRetryHandling()
{
    AIRequestQueue queue;
    DummyReply reply1;
    DummyReply reply2;

    queue.enqueue(makeRequest("queued"));
    queue.registerReply(&reply1, makeRequest("active", 4));
    QVERIFY(queue.hasPending());
    QVERIFY(queue.hasWork());
    QVERIFY(queue.hasReply(&reply1));
    QVERIFY(!queue.hasReply(&reply2));
    AICompletionRequest blocked;
    QVERIFY(!queue.takeNext(blocked)); // pending request blocks the queue

    QCOMPARE(queue.pendingReplies().size(), 1);
    QVERIFY(queue.pendingReplies().first() == &reply1);
    QCOMPARE(queue.insertIndexForSession("active"), 4);

    AICompletionRequest active = queue.takeReply(&reply1);
    QCOMPARE(active.sessionId, QStringLiteral("active"));
    QVERIFY(!queue.hasReply(&reply1));

    queue.prepend(makeRequest("retry"));
    const QList<AICompletionRequest> pending = queue.takeAllPending();
    QVERIFY(pending.isEmpty());
    QVERIFY(queue.hasQueued());

    queue.registerReply(&reply2, makeRequest("next"));
    const QList<AICompletionRequest> allPending = queue.takeAllPending();
    QCOMPARE(allPending.size(), 1);
    QCOMPARE(allPending.first().sessionId, QStringLiteral("next"));
    QCOMPARE(queue.pendingReplies().size(), 0);
}

void TestAIRequestQueue::sessionRemovalAndClear()
{
    AIRequestQueue queue;
    queue.enqueue(makeRequest("keep"));
    queue.enqueue(makeRequest("drop"));
    queue.registerReply(nullptr, makeRequest("pending"));
    QVERIFY(queue.hasRequestForSession("pending"));

    queue.removeQueuedForSession("drop");
    QVERIFY(queue.hasRequestForSession("keep"));
    QVERIFY(!queue.hasRequestForSession("drop"));
    QVERIFY(queue.hasRequestForSession("pending"));

    const QList<AICompletionRequest> queued = queue.takeAllQueued();
    QCOMPARE(queued.size(), 1);
    QCOMPARE(queued.first().sessionId, QStringLiteral("keep"));
    QVERIFY(!queue.hasQueued());
    QVERIFY(queue.hasPending());

    queue.clear();
    QVERIFY(!queue.hasWork());
    QVERIFY(!queue.hasRequestForSession("pending"));
}

QTEST_MAIN(TestAIRequestQueue)
#include "TestAIRequestQueue.moc"
