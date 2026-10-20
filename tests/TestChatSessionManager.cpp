#include <QtTest>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

#include "ChatSessionManager.h"

/**
 * @brief ChatSessionManager contract: current-session rules, attachment
 *        cleanup on remove() and the ChatSessionStore round trip.
 *
 * The manager is the only place that owns the session list, so these rules are
 * verified here instead of through AIAssistant (which needs a network stack and
 * a server process).
 */
class TestChatSessionManager : public QObject {
    Q_OBJECT

private slots:
    void createMakesNewSessionCurrent();
    void setCurrentRejectsUnknownId();
    void removeDeletesAttachmentsAndFallsBack();
    void removingLastSessionLeavesNoCurrent();
    void replaceAllSelectsLastSession();
    void saveAndLoadRoundTrip();
    void legacyPlaintextHistoryStillLoads();
    void unreadableHistoryYieldsNoSessions();
    void missingHistoryYieldsNoSessions();
    void corruptHistoryYieldsNoSessions();

private:
    static bool writeFile(const QString &path) {
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) return false;
        f.write("x");
        return true;
    }

    // The stored history must be sealed: the DPAPI blob ("enc1:") on Windows,
    // the vault's explicitly marked fallback ("plain:") elsewhere. Either way
    // the raw file must never carry the readable message text.
    static void verifySealed(const QByteArray &raw, const QString &secret) {
        QVERIFY(!raw.isEmpty());
#ifdef Q_OS_WIN
        QVERIFY(raw.startsWith("enc1:"));
        QVERIFY(!raw.contains(secret.toUtf8()));
#else
        QVERIFY(raw.startsWith("plain:"));
        QVERIFY(raw.contains(secret.toUtf8()));
#endif
    }
};

void TestChatSessionManager::createMakesNewSessionCurrent()
{
    ChatSessionManager sessions;
    QVERIFY(sessions.isEmpty());
    QVERIFY(sessions.currentId().isEmpty());
    QVERIFY(sessions.current() == nullptr);

    ChatSession &first = sessions.create("First");
    QVERIFY(!first.id.isEmpty());
    QVERIFY(first.createdAt.isValid());
    QCOMPARE(sessions.sessions().size(), 1);
    QCOMPARE(sessions.currentId(), first.id);
    QVERIFY(sessions.current() == &sessions.sessions().first());

    const QString firstId = first.id;
    ChatSession &second = sessions.create("Second");
    QCOMPARE(sessions.sessions().size(), 2);
    QCOMPARE(sessions.currentId(), second.id);
    QCOMPARE(sessions.find(firstId)->title, QString("First"));
    QVERIFY(sessions.find("missing") == nullptr);
}

void TestChatSessionManager::setCurrentRejectsUnknownId()
{
    ChatSessionManager sessions;
    const QString id = sessions.create("Only").id;
    sessions.create("Other");

    QVERIFY(!sessions.setCurrent("missing"));
    QCOMPARE(sessions.currentId(), sessions.sessions().last().id);

    QVERIFY(sessions.setCurrent(id));
    QCOMPARE(sessions.currentId(), id);
    QCOMPARE(sessions.current()->title, QString("Only"));
}

void TestChatSessionManager::removeDeletesAttachmentsAndFallsBack()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QVERIFY(QDir().mkpath(dir.path() + "/Upload"));
    QVERIFY(QDir().mkpath(dir.path() + "/Thumbnails"));
    const QString image = dir.path() + "/Upload/photo.png";
    const QString imageThumb = dir.path() + "/Thumbnails/photo.png";
    const QString doc = dir.path() + "/Upload/notes.txt";
    const QString docThumb = dir.path() + "/Thumbnails/notes.txt.png";
    QVERIFY(writeFile(image));
    QVERIFY(writeFile(imageThumb));
    QVERIFY(writeFile(doc));
    QVERIFY(writeFile(docThumb));

    ChatSessionManager sessions;
    const QString firstId = sessions.create("First").id;
    ChatSession &current = sessions.create("Second");
    current.messages.append(QJsonObject{
        {"role", "user"},
        {"attachments", QJsonArray{QString(image), QString(doc)}},
    });

    QVERIFY(sessions.remove(current.id));
    QCOMPARE(sessions.sessions().size(), 1);
    // The removed session was current → fall back to the last remaining one.
    QCOMPARE(sessions.currentId(), firstId);
    // Attachment files (upload + thumbnail, image and non-image) are gone.
    QVERIFY(!QFile::exists(image));
    QVERIFY(!QFile::exists(imageThumb));
    QVERIFY(!QFile::exists(doc));
    QVERIFY(!QFile::exists(docThumb));
    QVERIFY(!sessions.remove("missing"));
}

void TestChatSessionManager::removingLastSessionLeavesNoCurrent()
{
    ChatSessionManager sessions;
    const QString id = sessions.create("Only").id;
    QVERIFY(sessions.remove(id));
    QVERIFY(sessions.isEmpty());
    QVERIFY(sessions.currentId().isEmpty());
    QVERIFY(sessions.current() == nullptr);
    QVERIFY(sessions.currentMessages().isEmpty());
}

void TestChatSessionManager::replaceAllSelectsLastSession()
{
    ChatSession a;
    a.id = "a";
    a.title = "A";
    ChatSession b;
    b.id = "b";
    b.title = "B";

    ChatSessionManager sessions;
    sessions.replaceAll({a, b});
    QCOMPARE(sessions.sessions().size(), 2);
    QCOMPARE(sessions.currentId(), QString("b"));

    sessions.replaceAll({});
    QVERIFY(sessions.currentId().isEmpty());
}

void TestChatSessionManager::saveAndLoadRoundTrip()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString configPath = dir.path() + "/settings.ini";

    {
        ChatSessionManager sessions;
        ChatSession &session = sessions.create("Round trip");
        // The marker contains characters that the base64 alphabet cannot
        // produce, so "not in the raw file" is a deterministic check.
        session.messages.append(QJsonObject{{"role", "user"}, {"content", "secret-payload-4711"}});
        sessions.save("alice", configPath);
    }

    QFile stored(dir.path() + "/chat_sessions_alice.json");
    QVERIFY(stored.exists());
    QVERIFY(stored.open(QIODevice::ReadOnly));
    const QByteArray raw = stored.readAll();
    stored.close();

    // The history is sealed at rest: the file must not expose the content.
    verifySealed(raw, "secret-payload-4711");

    ChatSessionManager restored;
    restored.load("alice", configPath);
    QCOMPARE(restored.sessions().size(), 1);
    QCOMPARE(restored.sessions().first().title, QString("Round trip"));
    QCOMPARE(restored.currentId(), restored.sessions().first().id);
    const QList<QJsonObject> messages = restored.currentMessages();
    QCOMPARE(messages.size(), 1);
    QCOMPARE(messages.first().value("content").toString(), QString("secret-payload-4711"));

    // Unknown user → empty manager, no current session.
    ChatSessionManager other;
    other.load("bob", configPath);
    QVERIFY(other.isEmpty());
    QVERIFY(other.currentId().isEmpty());
}

void TestChatSessionManager::legacyPlaintextHistoryStillLoads()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString configPath = dir.path() + "/settings.ini";
    const QString historyPath = dir.path() + "/chat_sessions_alice.json";

    // A history written before the vault existed: readable JSON on disk.
    QFile legacy(historyPath);
    QVERIFY(legacy.open(QIODevice::WriteOnly));
    legacy.write("[{\"id\":\"legacy-id\",\"title\":\"Legacy\",\"createdAt\":\"2026-01-01T00:00:00\","
                 "\"titleIsDefault\":false,\"messages\":[{\"role\":\"user\",\"content\":\"old message\"}]}]");
    legacy.close();

    // It still loads (the vault passes legacy plaintext through unchanged).
    ChatSessionManager restored;
    restored.load("alice", configPath);
    QCOMPARE(restored.sessions().size(), 1);
    QCOMPARE(restored.sessions().first().title, QString("Legacy"));
    QCOMPARE(restored.sessions().first().messages.size(), 1);
    QCOMPARE(restored.sessions().first().messages.first().value("content").toString(),
             QString("old message"));

    // The next save() migrates it in place: same file name, sealed content.
    restored.save("alice", configPath);
    QFile resealed(historyPath);
    QVERIFY(resealed.open(QIODevice::ReadOnly));
    const QByteArray raw = resealed.readAll();
    resealed.close();
    verifySealed(raw, "old message");
}

void TestChatSessionManager::unreadableHistoryYieldsNoSessions()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString configPath = dir.path() + "/settings.ini";
    const QString historyPath = dir.path() + "/chat_sessions_alice.json";

    // A blob that cannot be unsealed (other user/machine): the store must
    // report an empty history instead of surfacing garbage.
    QFile foreign(historyPath);
    QVERIFY(foreign.open(QIODevice::WriteOnly));
    foreign.write("enc1:bm90LWEtdmFsaWQtYmxvYg==");
    foreign.close();

    ChatSessionManager restored;
    restored.load("alice", configPath);
    QVERIFY(restored.isEmpty());
    QVERIFY(restored.currentId().isEmpty());
    QVERIFY(restored.currentMessages().isEmpty());
}

void TestChatSessionManager::missingHistoryYieldsNoSessions()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString configPath = dir.path() + "/settings.ini";

    // No file at all: an empty history, no crash.
    ChatSessionManager restored;
    restored.load("alice", configPath);
    QVERIFY(restored.isEmpty());
    QVERIFY(restored.currentId().isEmpty());
    QVERIFY(restored.currentMessages().isEmpty());
}

void TestChatSessionManager::corruptHistoryYieldsNoSessions()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString configPath = dir.path() + "/settings.ini";
    const QString historyPath = dir.path() + "/chat_sessions_alice.json";

    // A file that is neither a sealed blob nor a session list.
    QFile corrupt(historyPath);
    QVERIFY(corrupt.open(QIODevice::WriteOnly));
    corrupt.write("this is not a session list");
    corrupt.close();

    ChatSessionManager restored;
    restored.load("alice", configPath);
    QVERIFY(restored.isEmpty());
    QVERIFY(restored.currentId().isEmpty());
    QVERIFY(restored.currentMessages().isEmpty());
}

QTEST_MAIN(TestChatSessionManager)
#include "TestChatSessionManager.moc"
