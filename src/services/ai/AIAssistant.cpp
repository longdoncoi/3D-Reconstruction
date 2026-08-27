#include "AIAssistant.h"
#include "AgentActionManifest.h"
#include "ChatSessionStore.h"
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

    QString sp = AppConfig::instance().aiAssistantDir() + "/" + AppConstants::AIServer::chatbotScript();
    if (QFileInfo::exists(sp)) {
        emit serverStatusChanged(LM_TR("ai.starting_server"));
        
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("PYTHONUNBUFFERED", "1");
        env.insert("PYTHONIOENCODING", "utf-8");
        env.insert("APP_DATA_DIR", QStandardPaths::writableLocation(QStandardPaths::AppDataLocation));
        aiServerProcess->setProcessEnvironment(env);
        
        aiServerProcess->setWorkingDirectory(AppConfig::instance().aiAssistantDir());
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
        // Keep the detached StartChatbotServer.py process running after Qt exits.
        aiServerProcess->terminate();
        if (!aiServerProcess->waitForFinished(1000)) {
            aiServerProcess->kill();
            aiServerProcess->waitForFinished(AppConstants::AIServer::STOP_SERVER_TIMEOUT_MS);
        }
    }

}

void AIAssistant::switchModel(int index) {
    startServer(index);
}

void AIAssistant::restartModel() {
    if (!isServerRunning()) {
        emit errorOccurred(LM_TR("ai.server_not_running"));
        return;
    }
    emit serverStatusChanged(tr("Đang tải lại Model..."));
    QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("reload-model"))};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(600000); // 10 min (model load có thể lâu)
    QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const auto doc = QJsonDocument::fromJson(reply->readAll());
            const QString msg = doc.object().value("message").toString("Model reloaded");
            emit serverStatusChanged(tr("✅ ") + msg);
        } else {
            emit errorOccurred(tr("Restart Model thất bại: ") + reply->errorString());
        }
    });
}

void AIAssistant::restartRAG() {
    if (!isServerRunning()) {
        emit errorOccurred(LM_TR("ai.server_not_running"));
        return;
    }
    emit serverStatusChanged(tr("Đang tải lại RAG index..."));
    QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("reload-rag"))};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(600000); // RAG rebuild can take several minutes on a large project.
    QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const auto doc = QJsonDocument::fromJson(reply->readAll());
            const QString msg = doc.object().value("message").toString("RAG reloaded");
            emit serverStatusChanged(tr("✅ ") + msg);
        } else {
            emit errorOccurred(tr("Restart RAG thất bại: ") + reply->errorString());
        }
    });
}

void AIAssistant::restartAgent() {
    if (!isServerRunning()) {
        emit errorOccurred(LM_TR("ai.server_not_running"));
        return;
    }
    emit serverStatusChanged(tr("Đang reset Agent..."));
    QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("reload-agent"))};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(15000);
    QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const auto doc = QJsonDocument::fromJson(reply->readAll());
            const QString msg = doc.object().value("message").toString("Agent reset");
            emit serverStatusChanged(tr("✅ ") + msg);
        } else {
            emit errorOccurred(tr("Restart Agent thất bại: ") + reply->errorString());
        }
    });
}

void AIAssistant::restartServer() {
    emit serverStatusChanged(tr("Đang khởi động lại AI Server..."));

    auto killManagerAndRestart = [this]() {
        // Discard anything in flight — the backend behind it is gone/going.
        m_queuedRequests.clear();
        const auto activeReplies = m_pendingRequests.keys();
        m_pendingRequests.clear();
        for (QNetworkReply *reply : activeReplies) {
            if (reply) reply->abort();
        }
        m_isThinking = false;
        m_serverReadyEmitted = false;
        m_serverRecoveryAttempts = 0;

        // Kill aiServerProcess — the ServerManager wrapper (server_manager.py).
        disconnect(aiServerProcess, nullptr, this, nullptr);
        if (aiServerProcess->state() != QProcess::NotRunning) {
            aiServerProcess->terminate();
            if (!aiServerProcess->waitForFinished(1000)) {
                aiServerProcess->kill();
                aiServerProcess->waitForFinished(AppConstants::AIServer::STOP_SERVER_TIMEOUT_MS);
            }
        }

        // Give the OS a moment to free port 8080 before the new ServerManager
        // instance probes /health and (re)spawns a fresh AI Server.
        QTimer::singleShot(800, this, [this]() {
            startServerProcess(m_currentModelIndex);
        });
    };

    if (isServerRunning()) {
        // Ask the AI Server that ServerManager spawned (detached, so we
        // don't otherwise hold a handle to it) to terminate itself first.
        QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("shutdown"))};
        req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        req.setTransferTimeout(5000);
        QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
        connect(reply, &QNetworkReply::finished, this, [reply, killManagerAndRestart]() {
            reply->deleteLater();
            killManagerAndRestart();   // run regardless of success/failure
        });
    } else {
        killManagerAndRestart();
    }
}

// ── Session management ────────────────────────────────────────────────────────

void AIAssistant::newChat() {
    ChatSession session;
    session.id        = ChatSessionStore::generateSessionId();
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

int AIAssistant::sessionThinkingInsertIndex(const QString &sessionId) const {
    for (auto it = m_pendingRequests.begin(); it != m_pendingRequests.end(); ++it) {
        if (it.value().sessionId == sessionId) {
            return it.value().insertAfterIndex;
        }
    }
    for (const auto &req : m_queuedRequests) {
        if (req.sessionId == sessionId) {
            return req.insertAfterIndex;
        }
    }
    return -1;
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
    for (const auto &m : messages) {
        // ``assistant_agent`` is UI-only state and is not a valid chat role.
        // Sending it to FastAPI was the source of HTTP 422 after changing mode.
        const QString role = m.value("role").toString();
        if (role == "user" || role == "assistant" || role == "system") msgs.append(m);
    }

    QJsonObject js;
    js["messages"]    = msgs;

    auto *userManager = UserManager::instance();
    double temp = userManager ? userManager->getUserPref(userManager->currentUsername(), "ai_temperature", QString::number(AppConstants::AIServer::DEFAULT_TEMPERATURE)).toDouble() : AppConstants::AIServer::DEFAULT_TEMPERATURE;
    js["temperature"] = temp;
    js["max_tokens"]  = AppConstants::AIServer::DEFAULT_MAX_TOKENS;
    js["language"] = LanguageManager::instance().currentLanguage();
    return js;
}

QJsonObject AIAssistant::buildAgentPayload(const QList<QJsonObject> &messages, const QString &task,
                                            const QStringList &attachments) const {
    QJsonArray history;
    // The current user message is passed as ``task``. Keep only the portable
    // conversational roles so Agent step JSON never contaminates model input.
    for (int index = 0; index < messages.size() - 1; ++index) {
        const QJsonObject &message = messages.at(index);
        const QString role = message.value("role").toString();
        if (role != "user" && role != "assistant") continue;
        QJsonObject item;
        item["role"] = role;
        item["content"] = message.value("content").toString();
        if (message.contains("attachments")) item["attachments"] = message.value("attachments");
        history.append(item);
    }

    auto *userManager = UserManager::instance();
    const double temperature = userManager
        ? userManager->getUserPref(userManager->currentUsername(), "ai_temperature", "0.3").toDouble()
        : 0.3;
    QJsonObject payload;
    payload["task"] = task;
    payload["history"] = history;
    payload["temperature"] = temperature;
    payload["language"] = LanguageManager::instance().currentLanguage();
    if (!attachments.isEmpty()) {
        QJsonArray values;
        for (const QString &attachment : attachments) values.append(attachment);
        payload["attachments"] = values;
    }
    return payload;
}

bool AIAssistant::isRecoverableConnectionError(QNetworkReply::NetworkError error) const {
    return error == QNetworkReply::ConnectionRefusedError ||
           error == QNetworkReply::RemoteHostClosedError ||
           error == QNetworkReply::NetworkSessionFailedError;
}

void AIAssistant::appendAssistantMessage(const QString &sessionId, const QString &content, int insertAfterIndex) {
    QJsonObject am;
    am["role"]      = "assistant";
    am["content"]   = content;
    am["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());

    ChatSession *sess = getSession(sessionId);
    if (sess) {
        if (insertAfterIndex >= 0 && insertAfterIndex < sess->messages.size()) {
            sess->messages.insert(insertAfterIndex + 1, am);
        } else {
            sess->messages.append(am);
        }
        saveAllSessions();
    }
}

void AIAssistant::enqueueCompletionRequest(const QString &sessionId, const QJsonObject &payload, int insertAfterIndex) {
    QueuedCompletionRequest request;
    request.sessionId = sessionId;
    request.payload = payload;
    request.insertAfterIndex = insertAfterIndex;
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

    QString urlStr = AppConstants::AIServer::apiEndpoint();
    if (request.isAgent) {
        if (request.isUiActionAck) {
            urlStr = urlStr.replace("/chat/completions", "/agent/ui-action-result");
        } else if (request.isApproval) {
            urlStr = urlStr.replace("/chat/completions", "/agent/approve");
        } else {
            urlStr = urlStr.replace("/chat/completions", "/agent/execute");
        }
    }
    QNetworkRequest req{QUrl(urlStr)};
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

    // Update timestamp of the message to reflect when retry occurred
    QJsonObject msg = sess->messages[msgIndex];
    msg["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    sess->messages[msgIndex] = msg;

    // Remove old response(s) immediately following the retried message
    while (sess->messages.size() > msgIndex + 1) {
        const QString nextRole = sess->messages[msgIndex + 1]["role"].toString();
        if (nextRole == "assistant" || nextRole == "assistant_agent") {
            sess->messages.removeAt(msgIndex + 1);
        } else {
            break;
        }
    }

    saveAllSessions();
    enqueueCompletionRequest(sessionId, buildCompletionPayload(sess->messages), msgIndex);
    emit historyChanged();
}

void AIAssistant::retryAgentTask(const QString &sessionId, int msgIndex) {
    if (hasPendingRequestForSession(sessionId)) return;

    ChatSession *sess = getSession(sessionId);
    if (!sess || msgIndex < 0 || msgIndex >= sess->messages.size()) return;

    const QJsonObject target = sess->messages[msgIndex];
    if (target["role"].toString() != "user" || target["content"].toString().trimmed().isEmpty()) return;

    QJsonObject msg = sess->messages[msgIndex];
    msg["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    sess->messages[msgIndex] = msg;

    // Remove old response(s) immediately following the retried message
    while (sess->messages.size() > msgIndex + 1) {
        const QString nextRole = sess->messages[msgIndex + 1]["role"].toString();
        if (nextRole == "assistant" || nextRole == "assistant_agent") {
            sess->messages.removeAt(msgIndex + 1);
        } else {
            break;
        }
    }

    saveAllSessions();

    QueuedCompletionRequest request;
    request.sessionId = sessionId;
    request.isAgent = true;
    request.insertAfterIndex = msgIndex;
    QList<QJsonObject> conversation;
    for (int index = 0; index <= msgIndex; ++index) conversation.append(sess->messages.at(index));
    request.payload = buildAgentPayload(conversation, msg["content"].toString(), {});
    request.payload["session_id"] = sessionId;
    m_queuedRequests.append(request);
    m_isThinking = true;
    processNextQueuedRequest();
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

// ── Agent mode ────────────────────────────────────────────────────────────────

void AIAssistant::executeAgentTask(const QString &sessionId, const QString &task,
                                   const QStringList &attachments) {
    if (task.isEmpty() && attachments.isEmpty()) return;
    const QString effectiveTask = task.isEmpty()
        ? QStringLiteral("Please analyze the attached files.")
        : task;
    
    QString targetSessionId = sessionId;
    ChatSession *sess = getSession(targetSessionId);
    if (!sess) {
        newChat();
        sess = currentSession();
        if (!sess) return;
        targetSessionId = sess->id;
    }

    if (hasPendingRequestForSession(targetSessionId)) return;

    QJsonObject um;
    um["role"]      = "user";
    um["content"]   = effectiveTask;
    um["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    
    if (!attachments.isEmpty()) {
        QJsonArray values;
        for (const QString &attachment : attachments) values.append(attachment);
        um["attachments"] = values;
    }
    sess->messages.append(um);

    if (sess->messages.size() == 1) {
        sess->title = effectiveTask.left(AppConstants::Chat::SESSION_TITLE_MAX_LENGTH) + (effectiveTask.length() > AppConstants::Chat::SESSION_TITLE_MAX_LENGTH ? "..." : "");
        emit sessionsChanged();
    }
    
    saveAllSessions();
    emit historyChanged();

    QJsonObject payload = buildAgentPayload(sess->messages, effectiveTask, attachments);
    payload["session_id"] = targetSessionId;

    QueuedCompletionRequest request;
    request.sessionId = targetSessionId;
    request.payload = payload;
    request.isAgent = true;
    request.isApproval = false;

    m_queuedRequests.append(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

void AIAssistant::approveAgentAction(const QString &sessionId, const QString &actionId) {
    QJsonObject payload;
    payload["action_id"] = actionId;
    payload["approved"] = true;
    payload["session_id"] = sessionId;

    QueuedCompletionRequest request;
    request.sessionId = sessionId;
    request.payload = payload;
    request.isAgent = true;
    request.isApproval = true;

    m_queuedRequests.append(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

void AIAssistant::rejectAgentAction(const QString &sessionId, const QString &actionId) {
    QJsonObject payload;
    payload["action_id"] = actionId;
    payload["approved"] = false;
    payload["session_id"] = sessionId;

    QueuedCompletionRequest request;
    request.sessionId = sessionId;
    request.payload = payload;
    request.isAgent = true;
    request.isApproval = true;

    m_queuedRequests.append(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

void AIAssistant::reportUiActionResult(const QString &requestId, bool success, const QVariantMap &result) {
    if (requestId.isEmpty()) return;
    QueuedCompletionRequest request;
    const auto entry = m_pendingUiActionSessions.take(requestId);
    request.sessionId = entry.first.isEmpty() ? m_currentSessionId : entry.first;
    request.insertAfterIndex = entry.second;  // -1 when not a retry (normal append)
    request.isAgent = true;
    request.isUiActionAck = true;
    request.payload["request_id"] = requestId;
    request.payload["success"] = success;
    request.payload["result"] = QJsonObject::fromVariantMap(result);
    m_queuedRequests.append(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

// ── Process callbacks ─────────────────────────────────────────────────────────

void AIAssistant::onProcessReadyRead() {
    QString out = QString::fromUtf8(aiServerProcess->readAllStandardOutput());
    qDebug().noquote() << "[AIAssistant Server]" << out;
    
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
    QString err = QString::fromUtf8(aiServerProcess->readAllStandardError());
    qDebug().noquote() << "[AIAssistant Server]" << err;
    
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
        if (request.isAgent) {
            QJsonObject res = QJsonDocument::fromJson(reply->readAll()).object();

            // application_action is deliberately executed by the desktop app,
            // not by the Python server. A successful ACK may carry one new,
            // sequential workflow action; dispatch only the request_id named
            // by this response, never an earlier step included in its history.
            const QString responseRequestId = res.value("request_id").toString();
            const bool hasFollowUpUiAction = request.isUiActionAck &&
                res.value("status").toString() == "pending_ui_action" &&
                !responseRequestId.isEmpty() &&
                responseRequestId != request.payload.value("request_id").toString();
            if (!request.isUiActionAck || hasFollowUpUiAction) {
                const QJsonArray steps = res.value("steps").toArray();
                for (const QJsonValue &value : steps) {
                    const QJsonObject step = value.toObject();
                    if (step.value("type").toString() != "tool_call" ||
                        step.value("tool").toString() != "application_action") {
                        continue;
                    }
                    QJsonObject params = step.value("params").toObject();
                    QString action = params.value("action").toString();
                    if (action.isEmpty()) continue;
                    const QString requestId = params.value("request_id").toString();
                    if (!responseRequestId.isEmpty() && requestId != responseRequestId) {
                        continue;
                    }
                    if (!requestId.isEmpty())
                        m_pendingUiActionSessions.insert(requestId, {sessionId, request.insertAfterIndex});
                    QString manifestError;
                    QVariantMap actionParams = params.toVariantMap();
                    if (!AgentActionManifest::canonicalize(action, actionParams, &manifestError)) {
                        reportUiActionResult(requestId, false, QVariantMap{{"error", manifestError}});
                        continue;
                    }
                    emit agentUiActionRequested(action, actionParams);
                }
            }
            
            // Append agent step message to history to keep it
            QJsonObject am;
            am["role"] = "assistant_agent";
            am["content"] = QString::fromUtf8(QJsonDocument(res).toJson(QJsonDocument::Compact));
            am["timestamp"] = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
            
            ChatSession *sess = getSession(sessionId);
            if (sess) {
                if (request.insertAfterIndex >= 0 && request.insertAfterIndex < sess->messages.size()) {
                    sess->messages.insert(request.insertAfterIndex + 1, am);
                } else {
                    sess->messages.append(am);
                }
                saveAllSessions();
            }
            
            if (res.value("status").toString() == "pending_approval") {
                emit agentApprovalRequired(sessionId, res);
            } else if (res.value("status").toString() != "pending_ui_action") {
                emit agentTaskCompleted(sessionId, "");
            }
            emit agentStepReceived(sessionId, res);
        } else {
            QJsonObject m = QJsonDocument::fromJson(reply->readAll())
                                .object()["choices"].toArray()[0]
                                .toObject()["message"].toObject();
            appendAssistantMessage(sessionId, m["content"].toString(), request.insertAfterIndex);
            emit responseReceived();
        }
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

        appendAssistantMessage(sessionId, errorMessage, request.insertAfterIndex);
    }
    
    processNextQueuedRequest();
    m_isThinking = hasPendingWork();
    emit historyChanged();
    
    reply->deleteLater();
}

// ── Persistence ───────────────────────────────────────────────────────────────

void AIAssistant::saveAllSessions() {
    ChatSessionStore::save(m_sessions);
}

void AIAssistant::loadAllSessions() {
    m_sessions = ChatSessionStore::load();
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
