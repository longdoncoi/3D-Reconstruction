#include "AIAssistant.h"
#include "AgentActionManifest.h"
#include "AppConstants.h"
#include "IAppPaths.h"
#include "IUserManager.h"
#include "ILanguageService.h"
#include <QDebug>
#include <QJsonDocument>
#include <QDateTime>

AIAssistant::AIAssistant(ILanguageService* lang, IUserManager* users,
                         IAppPaths* paths, QString pythonInterpreter,
                         int serverPort, QObject *parent,
                         QNetworkAccessManager* injectedNetwork)
    : IAIAssistantService(parent),
      m_lang(lang),
      m_users(users),
      m_paths(paths),
      m_pythonInterpreter(pythonInterpreter.isEmpty() ? QStringLiteral("python")
                                                      : pythonInterpreter),
      m_serverPort(serverPort),
      m_server(lang, paths, m_pythonInterpreter, nullptr),
      networkManager(injectedNetwork ? injectedNetwork
                                     : new QNetworkAccessManager(this)) {

    connect(&m_server, &AIServerProcessController::serverStatusChanged,
            this, [this](const QString &status) { emit serverStatusChanged(status); });
    connect(&m_server, &AIServerProcessController::errorOccurred,
            this, [this](const QString &message) { emit errorOccurred(message); });
    connect(&m_server, &AIServerProcessController::serverReady,
            this, &AIAssistant::markServerReady);
    connect(&m_server, &AIServerProcessController::processFinished,
            this, &AIAssistant::onServerFinished);
    connect(networkManager, &QNetworkAccessManager::finished, this, &AIAssistant::onReplyFinished);

    loadAllSessions(); // cũng chọn session cuối cùng làm hiện tại

    // If no sessions exist, create the first one
    if (m_sessions.isEmpty()) {
        newChat();
    }
}

AIAssistant::~AIAssistant() {
    stopServer();
}

// ── Server management ─────────────────────────────────────────────────────────

void AIAssistant::startServer(int modelIndex) {
    stopServer();
    m_serverRecoveryAttempts = 0;
    m_currentModelIndex = modelIndex;
    m_serverReadyEmitted = false;
    m_server.start(modelIndex);
}

void AIAssistant::resetTransientRequestState() {
    // Discard anything in flight — the backend behind it is gone/going.
    const auto activeReplies = m_requests.pendingReplies();
    m_requests.clear();
    for (QNetworkReply *reply : activeReplies) {
        if (reply) reply->abort();
    }
    // Pending UI-action ACKs belong to the dead server generation: without
    // this, a stale requestId could be answered by a restarted server and
    // replayed into the wrong session.
    m_pendingUiActionSessions.clear();
    m_isThinking = false;
    m_serverReadyEmitted = false;
    m_serverRecoveryAttempts = 0;
}

void AIAssistant::stopServer() {
    resetTransientRequestState();
    m_server.stop();
}

void AIAssistant::switchModel(int index) {
    const QStringList models = AppConstants::AIAssistant::modelNames();
    QString modelName = (index >= 0 && index < models.size()) ? models.at(index) : "Unknown Model";
    qDebug() << "[AI Server] Switching model to index:" << index << "(" << modelName << ")";
    startServer(index);
}

void AIAssistant::restartModel() {
    if (!isServerRunning()) {
        emit errorOccurred(m_lang->translate("ai.server_not_running"));
        return;
    }
    qDebug() << "[AI Server] Reloading current model";
    emit serverStatusChanged(m_lang->translate("ai.restarting_model"));
    QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("reload-model", m_serverPort))};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(600000); // 10 min (model load có thể lâu)
    QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const auto doc = QJsonDocument::fromJson(reply->readAll());
            const QString msg = doc.object().value("message").toString("Model reloaded");
            emit serverStatusChanged(QStringLiteral("✅ ") + msg);
        } else {
            emit errorOccurred(m_lang->translate("ai.restart_model_failed") + " " + reply->errorString());
        }
    });
}

void AIAssistant::restartRAG() {
    if (!isServerRunning()) {
        emit errorOccurred(m_lang->translate("ai.server_not_running"));
        return;
    }
    emit serverStatusChanged(m_lang->translate("ai.restarting_rag"));
    QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("reload-rag", m_serverPort))};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(600000); // RAG rebuild can take several minutes on a large project.
    QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const auto doc = QJsonDocument::fromJson(reply->readAll());
            const QString msg = doc.object().value("message").toString("RAG reloaded");
            emit serverStatusChanged(QStringLiteral("✅ ") + msg);
        } else {
            emit errorOccurred(m_lang->translate("ai.restart_rag_failed") + " " + reply->errorString());
        }
    });
}

void AIAssistant::restartAgent() {
    if (!isServerRunning()) {
        emit errorOccurred(m_lang->translate("ai.server_not_running"));
        return;
    }
    emit serverStatusChanged(m_lang->translate("ai.restarting_agent"));
    QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("reload-agent", m_serverPort))};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setTransferTimeout(15000);
    QNetworkReply *reply = networkManager->post(req, QByteArray("{}"));
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const auto doc = QJsonDocument::fromJson(reply->readAll());
            const QString msg = doc.object().value("message").toString("Agent reset");
            emit serverStatusChanged(QStringLiteral("✅ ") + msg);
        } else {
            emit errorOccurred(m_lang->translate("ai.restart_agent_failed") + " " + reply->errorString());
        }
    });
}

void AIAssistant::restartServer() {
    emit serverStatusChanged(m_lang->translate("ai.restarting_server"));

    auto killManagerAndRestart = [this]() {
        // Discard anything in flight — the backend behind it is gone/going.
        resetTransientRequestState();

        // Stop the server-manager wrapper. The controller owns the bounded
        // terminate/kill sequence and disconnects its own process signals.
        m_server.stop();

        // Give the OS a moment to free port 8080 before the new ServerManager
        // instance probes /health and (re)spawns a fresh AI Server.
        QTimer::singleShot(800, this, [this]() {
            m_server.start(m_currentModelIndex);
        });
    };

    if (isServerRunning()) {
        // Ask the AI Server that ServerManager spawned (detached, so we
        // don't otherwise hold a handle to it) to terminate itself first.
        QNetworkRequest req{QUrl(AppConstants::AIServer::adminEndpoint("shutdown", m_serverPort))};
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
    const QString title = m_lang->translate("ai.new_session") + " " +
                          QDateTime::currentDateTime().toString(AppConstants::Format::sessionDateTime());
    auto& session = m_sessions.create(title);
    session.titleIsDefault = true;
    saveAllSessions();
    emit sessionsChanged();
    emit historyChanged();
}

void AIAssistant::loadSession(const QString &sessionId) {
    if (m_sessions.setCurrent(sessionId)) {
        emit historyChanged();
    }
}

void AIAssistant::deleteSession(const QString &sessionId) {
    // Removes the session AND its attachment files; updates the current
    // selection (last remaining session, or none when the list is empty).
    if (!m_sessions.remove(sessionId)) return;

    m_requests.removeQueuedForSession(sessionId);
    m_isThinking = hasPendingWork();

    if (m_sessions.isEmpty()) newChat();
    saveAllSessions();
    emit sessionsChanged();
    emit historyChanged();
}

void AIAssistant::clearHistory() {
    newChat();
}

QList<QJsonObject> AIAssistant::getHistory() const {
    return m_sessions.currentMessages();
}

bool AIAssistant::isSessionThinking(const QString &sessionId) const {
    return hasPendingRequestForSession(sessionId);
}

int AIAssistant::sessionThinkingInsertIndex(const QString &sessionId) const {
    return m_requests.insertIndexForSession(sessionId);
}

ChatSession* AIAssistant::currentSession() {
    return m_sessions.current();
}

ChatSession* AIAssistant::getSession(const QString &sessionId) {
    return m_sessions.find(sessionId);
}

bool AIAssistant::hasPendingWork() const {
    return m_requests.hasWork();
}

bool AIAssistant::hasPendingRequestForSession(const QString &sessionId) const {
    return m_requests.hasRequestForSession(sessionId);
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

    auto *userManager = m_users;
    double temp = userManager ? userManager->getUserPref(userManager->currentUsername(), "ai_temperature", QString::number(AppConstants::AIServer::DEFAULT_TEMPERATURE)).toDouble() : AppConstants::AIServer::DEFAULT_TEMPERATURE;
    js["temperature"] = temp;
    js["max_tokens"]  = AppConstants::AIServer::DEFAULT_MAX_TOKENS;
    js["language"] = m_lang->currentLanguage();
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

    auto *userManager = m_users;
    const double temperature = userManager
        ? userManager->getUserPref(userManager->currentUsername(), "ai_temperature", "0.3").toDouble()
        : 0.3;
    QJsonObject payload;
    payload["task"] = task;
    payload["history"] = history;
    payload["temperature"] = temperature;
    payload["language"] = m_lang->currentLanguage();
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
    m_requests.enqueue(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

void AIAssistant::processNextQueuedRequest() {
    if (m_requests.hasPending() || !m_requests.hasQueued()) {
        m_isThinking = hasPendingWork();
        return;
    }

    if (!m_serverReadyEmitted) {
        m_isThinking = true;
        if (!m_server.isRunning()) {
            m_server.start(m_currentModelIndex);
        }
        return;
    }

    QueuedCompletionRequest request;
    if (!m_requests.takeNext(request)) return;

    QString baseUrl = AppConstants::AIServer::baseUrl(m_serverPort);
    QString urlStr;
    if (request.isAgent) {
        if (request.isUiActionAck) {
            urlStr = baseUrl + "/v1/agent/ui-action-result";
        } else if (request.isApproval) {
            urlStr = baseUrl + "/v1/agent/approve";
        } else {
            urlStr = baseUrl + "/v1/agent/execute";
        }
    } else {
        urlStr = baseUrl + "/v1/chat/completions";
    }
    QNetworkRequest req{QUrl(urlStr)};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    const int timeoutMs = (m_currentModelIndex == AppConstants::AIAssistant::VISION_MODEL_INDEX)
        ? AppConstants::AIServer::VISION_INFERENCE_TIMEOUT_MS
        : AppConstants::AIServer::TEXT_INFERENCE_TIMEOUT_MS;
    req.setTransferTimeout(timeoutMs);

    qDebug() << "[AI Server] req urlStr:" << urlStr;
    QNetworkReply *reply = networkManager->post(req, QJsonDocument(request.payload).toJson());
    m_requests.registerReply(reply, request);
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
            sess->title = m_lang->translate("ai.image_file");
        } else {
            sess->title = m_lang->translate("ai.new_chat");
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

    QStringList attachments;
    if (msg.contains("attachments")) {
        QJsonArray arr = msg["attachments"].toArray();
        for (int i = 0; i < arr.size(); ++i) {
            attachments.append(arr[i].toString());
        }
    }

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
    request.payload = buildAgentPayload(conversation, msg["content"].toString(), attachments);
    request.payload["session_id"] = sessionId;
    m_requests.enqueue(request);
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
    
    retryAgentTask(sessionId, msgIndex);
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

    m_requests.enqueue(request);
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

    m_requests.enqueue(request);
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

    m_requests.enqueue(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

void AIAssistant::reportUiActionResult(const QString &requestId, bool success, const QVariantMap &result) {
    if (requestId.isEmpty()) return;
    QueuedCompletionRequest request;
    const auto entry = m_pendingUiActionSessions.take(requestId);
    request.sessionId = entry.first.isEmpty() ? m_sessions.currentId() : entry.first;
    request.insertAfterIndex = entry.second;  // -1 when not a retry (normal append)
    request.isAgent = true;
    request.isUiActionAck = true;
    request.payload["request_id"] = requestId;
    request.payload["success"] = success;
    request.payload["result"] = QJsonObject::fromVariantMap(result);
    m_requests.enqueue(request);
    m_isThinking = true;
    processNextQueuedRequest();
}

// ── Process callbacks ─────────────────────────────────────────────────────────

void AIAssistant::markServerReady() {
    if (m_serverReadyEmitted) return;
    m_serverReadyEmitted = true;
    m_serverRecoveryAttempts = 0;
    emit serverStatusChanged(m_lang->translate("ai.server_ready"));
    processNextQueuedRequest();
    emit historyChanged();
}

void AIAssistant::onServerFinished(int exitCode, bool crashed) {
    m_serverReadyEmitted = false;

    if (hasPendingWork()) {
        const auto activeReplies = m_requests.pendingReplies();
        QList<QueuedCompletionRequest> activeRequests = m_requests.takeAllPending();

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
                m_requests.prepend(request);
            } else {
                appendAssistantMessage(request.sessionId, m_lang->translate("ai.connection_error"));
            }
        }

        m_isThinking = hasPendingWork();
        if (m_requests.hasQueued()) {
            if (m_serverRecoveryAttempts < 1) {
                m_serverRecoveryAttempts++;
                m_server.start(m_currentModelIndex);
            } else {
                const auto queuedRequests = m_requests.takeAllQueued();
                for (const QueuedCompletionRequest &request : queuedRequests) {
                    appendAssistantMessage(request.sessionId, m_lang->translate("ai.connection_error"));
                }
                m_isThinking = false;
            }
        }
        emit historyChanged();
        return;
    }

    if (crashed) {
        emit errorOccurred(m_lang->translate("ai.server_crashed"));
    } else if (exitCode != 0) {
        emit errorOccurred(m_lang->translate("ai.server_error").arg(exitCode));
    }
}

void AIAssistant::onReplyFinished(QNetworkReply* reply) {
    // Get the session ID for this specific reply
    if (!m_requests.hasReply(reply)) {
        reply->deleteLater();
        return;
    }
    
    QueuedCompletionRequest request = m_requests.takeReply(reply);
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
                    if (!requestId.isEmpty()) {
                        // Store the index where THIS response will be inserted, so the ACK is inserted AFTER it
                        int nextInsertIndex = (request.insertAfterIndex >= 0) ? request.insertAfterIndex + 1 : -1;
                        m_pendingUiActionSessions.insert(requestId, {sessionId, nextInsertIndex});
                    }
                    QString manifestError;
                    QVariantMap actionParams = params.toVariantMap();
                    if (!AgentActionManifest::canonicalize(action, actionParams, &manifestError, m_paths)) {
                        reportUiActionResult(requestId, false, QVariantMap{{"error", manifestError}});
                        continue;
                    }
                    // Agents send actions without file inputs; fill the demo
                    // fixtures from Config/agent_action_manifest.json so the
                    // plugins never embed sample paths themselves.
                    AgentActionManifest::applySampleDefaults(action, actionParams, m_paths);
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
            m_requests.prepend(request);
            m_serverReadyEmitted = false;

            if (!m_server.isRunning()) {
                m_server.start(m_currentModelIndex);
            }

            m_isThinking = true;
            emit historyChanged();
            reply->deleteLater();
            return;
        }

        QString errorMessage = m_lang->translate("ai.connection_error");
        if (reply->error() == QNetworkReply::TimeoutError) {
            errorMessage = m_lang->translate("ai.timeout_error");
        } else if (!responseBody.isEmpty()) {
            const QJsonObject errorObject = QJsonDocument::fromJson(responseBody).object();
            QString detail = errorObject.value("detail").toString();
            if (detail.isEmpty()) {
                detail = QString::fromUtf8(responseBody).trimmed();
            }
            if (!detail.isEmpty()) {
                errorMessage = m_lang->translate("ai.http_error").arg(detail.left(500));
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
    m_sessions.save(m_users ? m_users->currentUsername() : QString(),
                    m_paths ? m_paths->configPath() : QString());
}

void AIAssistant::loadAllSessions() {
    m_sessions.load(m_users ? m_users->currentUsername() : QString(),
                    m_paths ? m_paths->configPath() : QString());
}

void AIAssistant::reloadSessions() {
    loadAllSessions(); // selects the last stored session as current
    if (m_sessions.isEmpty()) {
        newChat();
    }
    emit sessionsChanged();
    emit historyChanged();
}
