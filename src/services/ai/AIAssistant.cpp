#include "AIAssistant.h"
#include "AppConfig.h"
#include "AppConstants.h"
#include "UserManager.h"
#include "LanguageManager.h"
#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QDateTime>
#include <QDir>
#include <QUuid>

AIAssistant::AIAssistant(QObject *parent) : QObject(parent) {
    aiServerProcess = new QProcess(this);
    networkManager = new QNetworkAccessManager(this);

    connect(aiServerProcess, &QProcess::readyReadStandardOutput, this, &AIAssistant::onProcessReadyRead);
    connect(aiServerProcess, &QProcess::readyReadStandardError, this, &AIAssistant::onProcessError);
    connect(aiServerProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &AIAssistant::onProcessFinished);
    connect(networkManager, &QNetworkAccessManager::finished, this, &AIAssistant::onReplyFinished);

    loadAllSessions();

    // If no sessions exist, create the first one
    if (m_sessions.isEmpty()) {
        newChat();
    } else {
        m_currentSessionId = m_sessions.last().id;
    }
}

AIAssistant::~AIAssistant() {
    stopServer();
}

// ── Server management ─────────────────────────────────────────────────────────

void AIAssistant::startServer(int modelIndex) {
    stopServer();
    m_serverReadyEmitted = false;
    
    // Đảm bảo kết nối lại tín hiệu cho tiến trình mới
    disconnect(aiServerProcess, nullptr, this, nullptr);
    connect(aiServerProcess, &QProcess::readyReadStandardOutput, this, &AIAssistant::onProcessReadyRead);
    connect(aiServerProcess, &QProcess::readyReadStandardError, this, &AIAssistant::onProcessError);
    connect(aiServerProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &AIAssistant::onProcessFinished);

    QString sp = AppConfig::instance().aiTrainingDir() + "/" + AppConstants::AIServer::chatbotScript();
    if (QFileInfo::exists(sp)) {
        emit serverStatusChanged(LM_TR("ai.starting_server"));
        
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("PYTHONUNBUFFERED", "1");
        aiServerProcess->setProcessEnvironment(env);
        
        aiServerProcess->start("python", QStringList() << "-u" << sp << QString::number(modelIndex));
    } else {
        emit errorOccurred(LM_TR("ai.missing_script"));
    }
}

void AIAssistant::stopServer() {
    // Ngắt toàn bộ kết nối để tránh nhận tín hiệu finished khi chủ động tắt
    disconnect(aiServerProcess, nullptr, this, nullptr);

    if (aiServerProcess->state() != QProcess::NotRunning) {
#ifdef Q_OS_WIN
        QProcess::execute("taskkill", QStringList() << "/F" << "/T" << "/PID"
                          << QString::number(aiServerProcess->processId()));
#else
        aiServerProcess->kill();
#endif
        aiServerProcess->waitForFinished(AppConstants::AIServer::STOP_SERVER_TIMEOUT_MS);
    }

    // Tiêu diệt triệt để bất kỳ tiến trình nào đang chiếm dụng cổng 8080 để tránh lỗi bind port
#ifdef Q_OS_WIN
    QProcess netstat;
    netstat.start("cmd", QStringList() << "/c" << QString("netstat -ano | findstr :%1").arg(AppConstants::AIServer::SERVER_PORT));
    if (netstat.waitForFinished(2000)) {
        QString output = QString::fromUtf8(netstat.readAllStandardOutput());
        QStringList lines = output.split('\n');
        for (const QString &line : lines) {
            if (line.contains("LISTENING")) {
                QStringList parts = line.simplified().split(' ');
                if (parts.size() >= 5) {
                    QString pid = parts.last();
                    bool ok;
                    int pidVal = pid.toInt(&ok);
                    if (ok && pidVal > 0) {
                        QProcess::execute("taskkill", QStringList() << "/F" << "/PID" << pid);
                    }
                }
            }
        }
    }
#else
    QProcess lsof;
    lsof.start("sh", QStringList() << "-c" << QString("lsof -t -i:%1").arg(AppConstants::AIServer::SERVER_PORT));
    if (lsof.waitForFinished(2000)) {
        QString output = QString::fromUtf8(lsof.readAllStandardOutput()).trimmed();
        if (!output.isEmpty()) {
            QProcess::execute("kill", QStringList() << "-9" << output);
        }
    }
#endif
}

void AIAssistant::switchModel(int index) {
    startServer(index);
}

// ── Session management ────────────────────────────────────────────────────────

void AIAssistant::newChat() {
    ChatSession session;
    session.id        = generateSessionId();
    session.title     = LM_TR("ai.new_session") + " " + QDateTime::currentDateTime().toString(AppConstants::Format::sessionDateTime());
    session.createdAt = QDateTime::currentDateTime();
    m_sessions.append(session);
    m_currentSessionId = session.id;
    saveAllSessions();
    emit sessionsChanged();
    emit historyChanged();
}

void AIAssistant::loadSession(const QString &sessionId) {
    for (const auto &s : m_sessions) {
        if (s.id == sessionId) {
            m_currentSessionId = sessionId;
            emit historyChanged();
            return;
        }
    }
}

void AIAssistant::deleteSession(const QString &sessionId) {
    bool found = false;
    for (int i = 0; i < m_sessions.size(); ++i) {
        if (m_sessions[i].id == sessionId) {
            // Physical cleanup of attachments
            for (const QJsonObject &msg : m_sessions[i].messages) {
                if (msg.contains("attachments")) {
                    QJsonArray atts = msg["attachments"].toArray();
                    for (const QJsonValue &v : atts) {
                        QString path = v.toString();
                        QFile::remove(path);
                        QString thumbPath = path;
                        thumbPath.replace("/Upload/", "/Thumbnails/");
                        if (!path.endsWith(".png") && !path.endsWith(".jpg") && !path.endsWith(".jpeg")) {
                            thumbPath += ".png";
                        }
                        QFile::remove(thumbPath);
                    }
                }
            }
            m_sessions.removeAt(i);
            found = true;
            break;
        }
    }
    if (!found) return;
    if (m_currentSessionId == sessionId) {
        m_currentSessionId = m_sessions.isEmpty() ? "" : m_sessions.last().id;
        if (m_currentSessionId.isEmpty()) newChat();
    }
    saveAllSessions();
    emit sessionsChanged();
    emit historyChanged();
}

void AIAssistant::clearHistory() {
    newChat();
}

QList<QJsonObject> AIAssistant::getHistory() const {
    for (const auto &s : m_sessions) {
        if (s.id == m_currentSessionId) return s.messages;
    }
    return {};
}

bool AIAssistant::isSessionThinking(const QString &sessionId) const {
    // Check if any pending request is for this session
    for (auto it = m_pendingRequests.begin(); it != m_pendingRequests.end(); ++it) {
        if (it.value() == sessionId) {
            return true;
        }
    }
    return false;
}

ChatSession* AIAssistant::currentSession() {
    for (auto &s : m_sessions) {
        if (s.id == m_currentSessionId) return &s;
    }
    return nullptr;
}

ChatSession* AIAssistant::getSession(const QString &sessionId) {
    for (auto &s : m_sessions) {
        if (s.id == sessionId) return &s;
    }
    return nullptr;
}

// ── Messaging ─────────────────────────────────────────────────────────────────

void AIAssistant::sendMessage(const QString &text, const QStringList &attachments) {
    ChatSession *sess = currentSession();
    if (!sess) { newChat(); sess = currentSession(); }
    sendMessageToSession(sess->id, text, attachments);
}

void AIAssistant::sendMessageToSession(const QString &sessionId, const QString &text, const QStringList &attachments) {
    if (text.isEmpty() && attachments.isEmpty()) return;
    ChatSession *sess = getSession(sessionId);
    if (!sess) return;

    QJsonObject um;
    um["role"]      = "user";
    um["content"]   = text;
    um["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    
    if (!attachments.isEmpty()) {
        QJsonArray attArray;
        for (const QString &att : attachments) {
            attArray.append(att);
        }
        um["attachments"] = attArray;
    }

    sess->messages.append(um);

    // Auto-title from first message
    if (sess->messages.size() == 1) {
        if (!text.isEmpty()) {
            sess->title = text.left(AppConstants::Chat::SESSION_TITLE_MAX_LENGTH) + (text.length() > AppConstants::Chat::SESSION_TITLE_MAX_LENGTH ? "..." : "");
        } else if (!attachments.isEmpty()) {
            sess->title = LM_TR("ai.image_file");
        } else {
            sess->title = LM_TR("ai.new_chat");
        }
        emit sessionsChanged();
    }

    saveAllSessions();
    emit historyChanged();

    QNetworkRequest req{QUrl(AppConstants::AIServer::apiEndpoint())};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QJsonArray msgs;
    for (const auto &m : sess->messages) msgs.append(m);

    QJsonObject js;
    js["messages"]    = msgs;
    js["temperature"] = AppConstants::AIServer::DEFAULT_TEMPERATURE;
    js["max_tokens"]  = AppConstants::AIServer::DEFAULT_MAX_TOKENS;

    // Set generous timeout for LLM inference (vision model can take 2-3 minutes)
    req.setTransferTimeout(AppConstants::AIServer::INFERENCE_TIMEOUT_MS);

    QNetworkReply *reply = networkManager->post(req, QJsonDocument(js).toJson());
    m_pendingRequests[reply] = sessionId;  // Track this request for the session
    m_isThinking = true;
    emit historyChanged();
}

// ── Process callbacks ─────────────────────────────────────────────────────────

void AIAssistant::onProcessReadyRead() {
    QString out = aiServerProcess->readAllStandardOutput();
    qDebug() << "[AIAssistant Server]" << out;
    
    QStringList lines = out.split('\n');
    for (const QString &line : lines) {
        if (line.contains("[STATUS]")) {
            int idx = line.indexOf("[STATUS]");
            QString statusText = line.mid(idx + 8).trimmed();
            emit serverStatusChanged(statusText);
        }
    }

    if (!m_serverReadyEmitted && (out.contains("[SUCCESS] AI Server") || out.contains("Uvicorn running on") || out.contains("Application startup complete"))) {
        m_serverReadyEmitted = true;
        emit serverStatusChanged(LM_TR("ai.server_ready"));
    }
}

void AIAssistant::onProcessError() {
    QString err = aiServerProcess->readAllStandardError();
    qDebug() << "[AIAssistant Server]" << err;
    
    QStringList lines = err.split('\n');
    for (const QString &line : lines) {
        if (line.contains("[STATUS]")) {
            int idx = line.indexOf("[STATUS]");
            QString statusText = line.mid(idx + 8).trimmed();
            emit serverStatusChanged(statusText);
        }
    }

    if (!m_serverReadyEmitted && (err.contains("[SUCCESS] AI Server") || err.contains("Uvicorn running on") || err.contains("Application startup complete"))) {
        m_serverReadyEmitted = true;
        emit serverStatusChanged(LM_TR("ai.server_ready"));
    }
}

void AIAssistant::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    if (exitStatus == QProcess::CrashExit) {
        emit errorOccurred(LM_TR("ai.server_crashed"));
    } else if (exitCode != 0) {
        emit errorOccurred(LM_TR("ai.server_error").arg(exitCode));
    }
}

void AIAssistant::onReplyFinished(QNetworkReply* reply) {
    // Get the session ID for this specific reply
    if (!m_pendingRequests.contains(reply)) {
        reply->deleteLater();
        return;
    }
    
    QString sessionId = m_pendingRequests.take(reply);  // Remove from pending and get sessionId
    
    if (reply->error() == QNetworkReply::NoError) {
        QJsonObject m = QJsonDocument::fromJson(reply->readAll())
                            .object()["choices"].toArray()[0]
                            .toObject()["message"].toObject();
        QJsonObject am;
        am["role"]      = "assistant";
        am["content"]   = m["content"].toString();
        am["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());

        ChatSession *sess = getSession(sessionId);
        if (sess) {
            sess->messages.append(am);
            saveAllSessions();
        }
        emit responseReceived();
        emit historyChanged();
    } else {
        emit errorOccurred(LM_TR("ai.connection_error"));
    }
    
    // Only set isThinking to false if no more pending requests
    if (m_pendingRequests.isEmpty()) {
        m_isThinking = false;
    }
    
    reply->deleteLater();
}

// ── Persistence ───────────────────────────────────────────────────────────────

void AIAssistant::saveAllSessions() {
    QJsonArray arr;
    for (const auto &s : m_sessions) {
        QJsonObject sObj;
        sObj["id"]        = s.id;
        sObj["title"]     = s.title;
        sObj["createdAt"] = s.createdAt.toString(Qt::ISODate);
        QJsonArray msgs;
        for (const auto &m : s.messages) msgs.append(m);
        sObj["messages"] = msgs;
        arr.append(sObj);
    }
    QFile f(getSessionsPath());
    if (f.open(QIODevice::WriteOnly)) {
        f.write(QJsonDocument(arr).toJson());
        f.close();
    }
}

void AIAssistant::loadAllSessions() {
    m_sessions.clear();
    QFile f(getSessionsPath());
    if (!f.open(QIODevice::ReadOnly)) return;
    QJsonArray arr = QJsonDocument::fromJson(f.readAll()).array();
    f.close();
    for (const auto &v : arr) {
        QJsonObject sObj = v.toObject();
        ChatSession s;
        s.id        = sObj["id"].toString();
        s.title     = sObj["title"].toString();
        s.createdAt = QDateTime::fromString(sObj["createdAt"].toString(), Qt::ISODate);
        for (const auto &m : sObj["messages"].toArray())
            s.messages.append(m.toObject());
        m_sessions.append(s);
    }
}

QString AIAssistant::getSessionsPath() {
    QString currentUser = UserManager::instance()->currentUsername();
    if (currentUser.isEmpty()) currentUser = "default";
    QString dir = QFileInfo(AppConfig::instance().configPath()).absolutePath() + "/";
    QDir().mkpath(dir);
    return dir + "chat_sessions_" + currentUser + ".json";
}

void AIAssistant::reloadSessions() {
    loadAllSessions();
    if (m_sessions.isEmpty()) {
        newChat();
    } else {
        m_currentSessionId = m_sessions.last().id;
    }
    emit sessionsChanged();
    emit historyChanged();
}

QString AIAssistant::generateSessionId() {
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}
