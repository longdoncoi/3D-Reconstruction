#include "AIAssistant.h"
#include <QStandardPaths>
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

AIAssistant::AIAssistant(QObject *parent)
    : IAIAssistantService(parent),
      aiServerProcess(new QProcess(this)),
      networkManager(new QNetworkAccessManager(this)) {

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
    m_serverRecoveryAttempts = 0;
    startServerProcess(modelIndex);
}

void AIAssistant::startServerProcess(int modelIndex) {
    m_currentModelIndex = modelIndex;
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
        env.insert("APP_DATA_DIR", QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/3D-Reconstruction");
        aiServerProcess->setProcessEnvironment(env);
        
        aiServerProcess->setWorkingDirectory(AppConfig::instance().aiTrainingDir());
        aiServerProcess->start("python", QStringList() << "-u" << sp << QString::number(modelIndex));
    } else {
        emit errorOccurred(LM_TR("ai.missing_script"));
    }
}

void AIAssistant::stopServer() {
    m_queuedRequests.clear();
    const auto activeReplies = m_pendingRequests.keys();
    m_pendingRequests.clear();
    for (QNetworkReply *reply : activeReplies) {
        if (reply) reply->abort();
    }
    m_isThinking = false;
    m_serverReadyEmitted = false;
    m_serverRecoveryAttempts = 0;

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

    for (int i = m_queuedRequests.size() - 1; i >= 0; --i) {
        if (m_queuedRequests[i].sessionId == sessionId) {
            m_queuedRequests.removeAt(i);
        }
    }
    m_isThinking = hasPendingWork();

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
    return hasPendingRequestForSession(sessionId);
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

bool AIAssistant::hasPendingWork() const {
    return !m_pendingRequests.isEmpty() || !m_queuedRequests.isEmpty();
}

bool AIAssistant::hasPendingRequestForSession(const QString &sessionId) const {
    for (auto it = m_pendingRequests.begin(); it != m_pendingRequests.end(); ++it) {
        if (it.value().sessionId == sessionId) return true;
    }

    for (const QueuedCompletionRequest &request : m_queuedRequests) {
        if (request.sessionId == sessionId) return true;
    }

    return false;
}

QJsonObject AIAssistant::buildCompletionPayload(const QList<QJsonObject> &messages) const {
    QJsonArray msgs;
    for (const auto &m : messages) msgs.append(m);

    QJsonObject js;
    js["messages"]    = msgs;
    js["temperature"] = AppConstants::AIServer::DEFAULT_TEMPERATURE;
    js["max_tokens"]  = AppConstants::AIServer::DEFAULT_MAX_TOKENS;
    return js;
}

bool AIAssistant::isRecoverableConnectionError(QNetworkReply::NetworkError error) const {
    return error == QNetworkReply::ConnectionRefusedError ||
           error == QNetworkReply::RemoteHostClosedError ||
           error == QNetworkReply::NetworkSessionFailedError;
}

void AIAssistant::appendAssistantMessage(const QString &sessionId, const QString &content) {
    QJsonObject am;
    am["role"]      = "assistant";
    am["content"]   = content;
    am["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());

    ChatSession *sess = getSession(sessionId);
    if (sess) {
        sess->messages.append(am);
        saveAllSessions();
    }
}

void AIAssistant::enqueueCompletionRequest(const QString &sessionId, const QJsonObject &payload) {
    QueuedCompletionRequest request;
    request.sessionId = sessionId;
    request.payload = payload;
    m_queuedRequests.append(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

void AIAssistant::processNextQueuedRequest() {
    if (!m_pendingRequests.isEmpty() || m_queuedRequests.isEmpty()) {
        m_isThinking = hasPendingWork();
        return;
    }

    if (!m_serverReadyEmitted) {
        m_isThinking = true;
        if (aiServerProcess->state() == QProcess::NotRunning) {
            startServerProcess(m_currentModelIndex);
        }
        return;
    }

    const QueuedCompletionRequest request = m_queuedRequests.takeFirst();

    QNetworkRequest req{QUrl(AppConstants::AIServer::apiEndpoint())};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    const int timeoutMs = (m_currentModelIndex == AppConstants::AIAssistant::VISION_MODEL_INDEX)
        ? AppConstants::AIServer::VISION_INFERENCE_TIMEOUT_MS
        : AppConstants::AIServer::TEXT_INFERENCE_TIMEOUT_MS;
    req.setTransferTimeout(timeoutMs);

    QNetworkReply *reply = networkManager->post(req, QJsonDocument(request.payload).toJson());
    m_pendingRequests[reply] = request;
    m_isThinking = true;
}

// ── Messaging ─────────────────────────────────────────────────────────────────

void AIAssistant::sendMessage(const QString &text, const QStringList &attachments) {
    ChatSession *sess = currentSession();
    if (!sess) { newChat(); sess = currentSession(); }
    sendMessageToSession(sess->id, text, attachments);
}

void AIAssistant::sendMessageToSession(const QString &sessionId, const QString &text, const QStringList &attachments) {
    if (text.isEmpty() && attachments.isEmpty()) return;
    if (hasPendingRequestForSession(sessionId)) return;

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
    enqueueCompletionRequest(sessionId, buildCompletionPayload(sess->messages));
    emit historyChanged();
}

void AIAssistant::retryMessage(const QString &sessionId, int msgIndex) {
    if (hasPendingRequestForSession(sessionId)) return;

    ChatSession *sess = getSession(sessionId);
    if (!sess) return;
    if (msgIndex < 0 || msgIndex >= sess->messages.size()) return;

    // Discard any messages after the target message
    while (sess->messages.size() > msgIndex + 1) {
        sess->messages.removeLast();
    }
    
    // Update timestamp of the message to reflect when retry occurred
    QJsonObject msg = sess->messages[msgIndex];
    msg["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    sess->messages[msgIndex] = msg;

    saveAllSessions();
    enqueueCompletionRequest(sessionId, buildCompletionPayload(sess->messages));
    emit historyChanged();
}

void AIAssistant::editMessage(const QString &sessionId, int msgIndex, const QString &newText) {
    ChatSession *sess = getSession(sessionId);
    if (!sess) return;
    if (msgIndex < 0 || msgIndex >= sess->messages.size()) return;
    
    QJsonObject msg = sess->messages[msgIndex];
    msg["content"] = newText;
    sess->messages[msgIndex] = msg;
    
    retryMessage(sessionId, msgIndex);
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
        m_serverRecoveryAttempts = 0;
        emit serverStatusChanged(LM_TR("ai.server_ready"));
        processNextQueuedRequest();
        emit historyChanged();
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
        m_serverRecoveryAttempts = 0;
        emit serverStatusChanged(LM_TR("ai.server_ready"));
        processNextQueuedRequest();
        emit historyChanged();
    }
}

void AIAssistant::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    m_serverReadyEmitted = false;

    if (hasPendingWork()) {
        const auto activeReplies = m_pendingRequests.keys();
        QList<QueuedCompletionRequest> activeRequests = m_pendingRequests.values();
        m_pendingRequests.clear();

        for (QNetworkReply *reply : activeReplies) {
            if (reply) {
                reply->abort();
                reply->deleteLater();
            }
        }

        for (int i = activeRequests.size() - 1; i >= 0; --i) {
            QueuedCompletionRequest request = activeRequests[i];
            if (request.retryCount < 1) {
                request.retryCount++;
                m_queuedRequests.prepend(request);
            } else {
                appendAssistantMessage(request.sessionId, LM_TR("ai.connection_error"));
            }
        }

        m_isThinking = hasPendingWork();
        if (!m_queuedRequests.isEmpty()) {
            if (m_serverRecoveryAttempts < 1) {
                m_serverRecoveryAttempts++;
                startServerProcess(m_currentModelIndex);
            } else {
                const auto queuedRequests = m_queuedRequests;
                m_queuedRequests.clear();
                for (const QueuedCompletionRequest &request : queuedRequests) {
                    appendAssistantMessage(request.sessionId, LM_TR("ai.connection_error"));
                }
                m_isThinking = false;
            }
        }
        emit historyChanged();
        return;
    }

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
    
    QueuedCompletionRequest request = m_pendingRequests.take(reply);
    QString sessionId = request.sessionId;  // Remove from pending and get sessionId
    
    if (reply->error() == QNetworkReply::NoError) {
        QJsonObject m = QJsonDocument::fromJson(reply->readAll())
                            .object()["choices"].toArray()[0]
                            .toObject()["message"].toObject();
        appendAssistantMessage(sessionId, m["content"].toString());
        emit responseReceived();
    } else {
        const QByteArray responseBody = reply->readAll();
        qWarning() << "[AIAssistant] Request failed:" << reply->errorString();

        if (isRecoverableConnectionError(reply->error()) && request.retryCount < 1) {
            request.retryCount++;
            m_queuedRequests.prepend(request);
            m_serverReadyEmitted = false;

            if (aiServerProcess->state() == QProcess::NotRunning) {
                startServerProcess(m_currentModelIndex);
            }

            m_isThinking = true;
            emit historyChanged();
            reply->deleteLater();
            return;
        }

        QString errorMessage = LM_TR("ai.connection_error");
        if (reply->error() == QNetworkReply::TimeoutError) {
            errorMessage = LM_TR("ai.timeout_error");
        } else if (!responseBody.isEmpty()) {
            const QJsonObject errorObject = QJsonDocument::fromJson(responseBody).object();
            QString detail = errorObject.value("detail").toString();
            if (detail.isEmpty()) {
                detail = QString::fromUtf8(responseBody).trimmed();
            }
            if (!detail.isEmpty()) {
                errorMessage = LM_TR("ai.http_error").arg(detail.left(500));
            }
        }

        appendAssistantMessage(sessionId, errorMessage);
    }
    
    processNextQueuedRequest();
    m_isThinking = hasPendingWork();
    emit historyChanged();
    
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
