#include "AIAssistant.h"
#include "AIAssistantPayloadBuilder.h"
#include "AppConstants.h"
#include "IAppPaths.h"
#include "IAuthService.h"
#include "IUserPreferences.h"
#include "ILanguageService.h"
#include <QDebug>
#include <QJsonDocument>
#include <QDateTime>

namespace {
/// The agent endpoint historically used a lower temperature than the plain
/// completion endpoint; keep that contract while the two payloads are built
/// in one place.
constexpr double kDefaultAgentTemperature = 0.3;
constexpr char kAuthorizationHeader[] = "Authorization";
} // namespace

void AIAssistant::authorizeRequest(QNetworkRequest &request) const
{
    request.setRawHeader(kAuthorizationHeader,
                         QByteArrayLiteral("Bearer ") + m_server.accessToken().toUtf8());
}

AIAssistant::AIAssistant(ILanguageService* lang, IAuthService* auth,
                         IUserPreferences* prefs, IAppPaths* paths,
                         QString pythonInterpreter, int serverPort,
                         QObject *parent, QNetworkAccessManager* injectedNetwork)
    : IAIAssistantService(parent),
      m_lang(lang),
      m_auth(auth),
      m_prefs(prefs),
      m_paths(paths),
      m_pythonInterpreter(pythonInterpreter.isEmpty() ? QStringLiteral("python")
                                                      : pythonInterpreter),
      m_serverPort(serverPort),
      m_server(lang, paths, m_pythonInterpreter, nullptr),
      networkManager(injectedNetwork ? injectedNetwork
                                     : new QNetworkAccessManager(this)),
      m_adminClient(lang, networkManager, serverPort,
                    [this]() { return m_server.accessToken(); }),
      m_agentActions(paths, this) {

    connect(&m_server, &AIServerProcessController::serverStatusChanged,
            this, [this](const QString &status) { emit serverStatusChanged(status); });
    connect(&m_server, &AIServerProcessController::errorOccurred,
            this, [this](const QString &message) { emit errorOccurred(message); });
    connect(&m_server, &AIServerProcessController::serverReady,
            this, &AIAssistant::markServerReady);
    connect(&m_server, &AIServerProcessController::processFinished,
            this, &AIAssistant::onServerFinished);
    connect(networkManager, &QNetworkAccessManager::finished, this, &AIAssistant::onReplyFinished);
    connect(&m_adminClient, &AIAdminClient::statusChanged,
            this, &IAIAssistantService::serverStatusChanged);
    connect(&m_adminClient, &AIAdminClient::errorOccurred,
            this, &IAIAssistantService::errorOccurred);
    connect(&m_agentActions, &AgentActionDispatcher::uiActionRequested,
            this, &IAIAssistantService::agentUiActionRequested);
    connect(&m_agentActions, &AgentActionDispatcher::uiActionRejected,
            this, [this](const QString &requestId, const QString &error) {
                reportUiActionResult(requestId, false, QVariantMap{{"error", error}});
            });

    m_agentStepTimer = new QTimer(this);
    m_agentStepTimer->setInterval(20);
    connect(m_agentStepTimer, &QTimer::timeout, this, &AIAssistant::onAgentStepTimerTick);

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
        if (reply) {
            clearStreamState(reply);
            reply->abort();
        }
    }
    // Pending UI-action ACKs belong to the dead server generation: without
    // this, a stale requestId could be answered by a restarted server and
    // replayed into the wrong session.
    m_agentActions.clearPending();
    m_isThinking = false;
    m_serverReadyEmitted = false;
    m_serverRecoveryAttempts = 0;
}

void AIAssistant::stopServer() {
    resetTransientRequestState();
    m_server.stop();
}

void AIAssistant::cancelSessionRequest(const QString &sessionId) {
    if (sessionId.isEmpty() || !m_requests.hasRequestForSession(sessionId)) return;

    // Remove local state before aborting: QNetworkReply::abort() completes the
    // reply, and the finished signal must not turn an intentional Stop into a
    // connection-error message or retry.
    const QList<QNetworkReply *> activeReplies =
        m_requests.takePendingRepliesForSession(sessionId);
    m_requests.removeQueuedForSession(sessionId);

    if (!activeReplies.isEmpty() && m_serverReadyEmitted) {
        QNetworkRequest request{
            QUrl(AppConstants::AIServer::baseUrl(m_serverPort) + "/v1/agent/cancel")};
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        authorizeRequest(request);
        request.setTransferTimeout(5000);
        QJsonObject payload;
        payload["session_id"] = sessionId;
        // This reply is intentionally not registered in AIRequestQueue: the
        // global finished handler treats it as transport-only and deletes it.
        networkManager->post(request, QJsonDocument(payload).toJson());
    }

    for (QNetworkReply *reply : activeReplies) {
        if (reply) {
            clearStreamState(reply);
            reply->abort();
        }
    }

    m_isThinking = hasPendingWork();
    processNextQueuedRequest();
    emit historyChanged();
}

void AIAssistant::switchModel(int index) {
    const QStringList models = AppConstants::AIAssistant::modelNames();
    QString modelName = (index >= 0 && index < models.size()) ? models.at(index) : "Unknown Model";
    qDebug() << "[AI Server] Switching model to index:" << index << "(" << modelName << ")";
    startServer(index);
}

void AIAssistant::restartModel() {
    m_adminClient.restartModel(isServerRunning());
}

void AIAssistant::restartRAG() {
    m_adminClient.restartRag(isServerRunning());
}

void AIAssistant::restartAgent() {
    m_adminClient.restartAgent(isServerRunning());
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
        authorizeRequest(req);
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

void AIAssistant::renameSession(const QString &sessionId, const QString &title) {
    if (!m_sessions.rename(sessionId, title)) return;

    saveAllSessions();
    emit sessionsChanged();
}

void AIAssistant::clearHistory() {
    newChat();
}

QList<ChatMessage> AIAssistant::getHistory() const {
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

double AIAssistant::userTemperature(double fallback) const {
    const QString username = m_auth ? m_auth->currentUsername() : QString();
    return m_prefs
        ? m_prefs->getUserPref(username, "ai_temperature", QString::number(fallback)).toDouble()
        : fallback;
}

bool AIAssistant::isRecoverableConnectionError(QNetworkReply::NetworkError error) const {
    return error == QNetworkReply::ConnectionRefusedError ||
           error == QNetworkReply::RemoteHostClosedError ||
           error == QNetworkReply::NetworkSessionFailedError;
}

void AIAssistant::appendAssistantMessage(const QString &sessionId, const QString &content, int insertAfterIndex) {
    ChatMessage am;
    am.role = ChatRole::Assistant;
    am.content = content;
    am.timestamp = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());

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
    const bool streamsResponse = !request.isAgent ||
        (!request.isApproval && !request.isUiActionAck);
    if (streamsResponse) {
        req.setRawHeader("Accept", "text/event-stream");
        req.setRawHeader("Connection", "close");
    }
    authorizeRequest(req);
    const int timeoutMs = (m_currentModelIndex == AppConstants::AIAssistant::VISION_MODEL_INDEX)
        ? AppConstants::AIServer::VISION_INFERENCE_TIMEOUT_MS
        : AppConstants::AIServer::TEXT_INFERENCE_TIMEOUT_MS;
    req.setTransferTimeout(timeoutMs);

    qDebug() << "[AI Server] req urlStr:" << urlStr;
    QNetworkReply *reply = networkManager->post(req, QJsonDocument(request.payload).toJson());
    m_requests.registerReply(reply, request);
    if (streamsResponse) {
        connect(reply, &QIODevice::readyRead, this,
                [this, reply]() { onReplyReadyRead(reply); });
    }
    m_isThinking = true;
}

// ── Messaging ─────────────────────────────────────────────────────────────────

bool AIAssistant::applyAutoTitle(ChatSession &session, const QString &text,
                                 const QStringList &attachments) {
    if (!session.titleIsDefault) return false;

    if (!text.isEmpty()) {
        session.title = text.left(AppConstants::Chat::SESSION_TITLE_MAX_LENGTH) + (text.length() > AppConstants::Chat::SESSION_TITLE_MAX_LENGTH ? "..." : "");
    } else if (!attachments.isEmpty()) {
        session.title = m_lang->translate("ai.image_file");
    } else {
        session.title = m_lang->translate("ai.new_chat");
    }
    // The placeholder has been replaced: clearing the flag is what makes the
    // sidebar show the prompt, and it keeps a later rename authoritative.
    session.titleIsDefault = false;
    return true;
}

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

    ChatMessage um;
    um.role = ChatRole::User;
    um.content = text;
    um.timestamp = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    um.attachments = attachments;

    sess->messages.append(um);

    // Auto-title from the first prompt while the title is still a placeholder.
    if (applyAutoTitle(*sess, text, attachments)) {
        emit sessionsChanged();
    }

    saveAllSessions();
    enqueueCompletionRequest(sessionId,
        AIAssistantPayloadBuilder::buildCompletionPayload(
            sess->messages,
            userTemperature(AppConstants::AIServer::DEFAULT_TEMPERATURE),
            m_lang->currentLanguage()));
    emit historyChanged();
}

void AIAssistant::retryMessage(const QString &sessionId, int msgIndex) {
    if (hasPendingRequestForSession(sessionId)) return;

    ChatSession *sess = getSession(sessionId);
    if (!sess) return;
    if (msgIndex < 0 || msgIndex >= sess->messages.size()) return;

    // Update timestamp of the message to reflect when retry occurred
    ChatMessage msg = sess->messages[msgIndex];
    msg.timestamp = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    sess->messages[msgIndex] = msg;

    // Remove old response(s) immediately following the retried message
    while (sess->messages.size() > msgIndex + 1) {
        const ChatRole nextRole = sess->messages[msgIndex + 1].role;
        if (nextRole == ChatRole::Assistant || nextRole == ChatRole::AgentTrace) {
            sess->messages.removeAt(msgIndex + 1);
        } else {
            break;
        }
    }

    saveAllSessions();
    enqueueCompletionRequest(sessionId,
        AIAssistantPayloadBuilder::buildCompletionPayload(
            sess->messages,
            userTemperature(AppConstants::AIServer::DEFAULT_TEMPERATURE),
            m_lang->currentLanguage()),
        msgIndex);
    emit historyChanged();
}

void AIAssistant::retryAgentTask(const QString &sessionId, int msgIndex) {
    if (hasPendingRequestForSession(sessionId)) return;

    ChatSession *sess = getSession(sessionId);
    if (!sess || msgIndex < 0 || msgIndex >= sess->messages.size()) return;

    const ChatMessage target = sess->messages[msgIndex];
    if (target.role != ChatRole::User || target.content.trimmed().isEmpty()) return;

    ChatMessage msg = sess->messages[msgIndex];
    msg.timestamp = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    sess->messages[msgIndex] = msg;

    const QStringList attachments = msg.attachments;

    // Remove old response(s) immediately following the retried message
    while (sess->messages.size() > msgIndex + 1) {
        const ChatRole nextRole = sess->messages[msgIndex + 1].role;
        if (nextRole == ChatRole::Assistant || nextRole == ChatRole::AgentTrace) {
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
    QList<ChatMessage> conversation;
    for (int index = 0; index <= msgIndex; ++index) conversation.append(sess->messages.at(index));
    request.payload = AIAssistantPayloadBuilder::buildAgentPayload(
        conversation, msg.content, attachments,
        userTemperature(kDefaultAgentTemperature), m_lang->currentLanguage());
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
    
    ChatMessage msg = sess->messages[msgIndex];
    msg.content = newText;
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

    ChatMessage um;
    um.role = ChatRole::User;
    um.content = effectiveTask;
    um.timestamp = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
    um.attachments = attachments;
    sess->messages.append(um);

    if (applyAutoTitle(*sess, effectiveTask, attachments)) {
        emit sessionsChanged();
    }
    
    saveAllSessions();

    QJsonObject payload = AIAssistantPayloadBuilder::buildAgentPayload(
        sess->messages, effectiveTask, attachments,
        userTemperature(kDefaultAgentTemperature), m_lang->currentLanguage());
    payload["session_id"] = targetSessionId;

    QueuedCompletionRequest request;
    request.sessionId = targetSessionId;
    request.payload = payload;
    request.isAgent = true;
    request.isApproval = false;

    m_requests.enqueue(request);
    m_isThinking = true;
    processNextQueuedRequest();
    // Emit AFTER the request is registered: updateChatUI() reads
    // isSessionThinking() to draw the "AI đang suy nghĩ..." marker, so the
    // signal sent before enqueue() used to render an idle chat.
    emit historyChanged();
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
    const auto entry = m_agentActions.takePending(requestId);
    request.sessionId = entry.first.isEmpty() ? m_sessions.currentId() : entry.first;
    request.insertAfterIndex = entry.second;  // -1 when not a retry (normal append)
    request.isAgent = true;
    request.isUiActionAck = true;
    request.payload["request_id"] = requestId;
    request.payload["success"] = success;
    request.payload["result"] = QJsonObject::fromVariantMap(result);
    // The server binds the ack to the session that parked the action (ADR
    // 0047); echo it exactly like ``approveAgentAction`` does so a queued or
    // stale ack cannot resume another session's run.
    request.payload["session_id"] = request.sessionId;
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

bool AIAssistant::isEventStream(QNetworkReply *reply) const {
    return reply && reply->header(QNetworkRequest::ContentTypeHeader)
                        .toString().startsWith(QStringLiteral("text/event-stream"));
}

void AIAssistant::updateStreamDraft(QNetworkReply *reply,
                                    const QueuedCompletionRequest &request,
                                    const QString &content) {
    ChatSession *session = getSession(request.sessionId);
    if (!session) return;

    int index = m_streamDraftIndices.value(reply, -1);
    if (index < 0) {
        ChatMessage message;
        message.role = ChatRole::Assistant;
        message.timestamp = QDateTime::currentDateTime().toString(
            AppConstants::Format::chatTimestamp());
        index = request.insertAfterIndex >= 0 && request.insertAfterIndex < session->messages.size()
            ? request.insertAfterIndex + 1
            : session->messages.size();
        session->messages.insert(index, message);
        m_streamDraftIndices.insert(reply, index);
    }
    if (index >= session->messages.size()) return;
    session->messages[index].content = content;
    emit historyChanged();
}

void AIAssistant::updateAgentStreamDraft(
    QNetworkReply *reply, const QueuedCompletionRequest &request,
    const QJsonObject &envelope) {
    ChatSession *session = getSession(request.sessionId);
    if (!session) return;

    int index = m_streamDraftIndices.value(reply, -1);
    if (index < 0) {
        ChatMessage message;
        message.role = ChatRole::AgentTrace;
        message.timestamp = QDateTime::currentDateTime().toString(
            AppConstants::Format::chatTimestamp());
        index = request.insertAfterIndex >= 0 && request.insertAfterIndex < session->messages.size()
            ? request.insertAfterIndex + 1
            : session->messages.size();
        session->messages.insert(index, message);
        m_streamDraftIndices.insert(reply, index);
    }
    if (index >= session->messages.size()) return;
    session->messages[index].content = QString::fromUtf8(
        QJsonDocument(envelope).toJson(QJsonDocument::Compact));
    emit historyChanged();
}

void AIAssistant::enqueueAgentStepForStreaming(
    QNetworkReply *reply,
    const QueuedCompletionRequest &request,
    const QJsonObject &step)
{
    StreamingStepState &st = m_agentStepStreamers[reply];
    st.request = request;
    st.pendingSteps.enqueue(step);
    if (m_agentStepTimer && !m_agentStepTimer->isActive()) {
        m_agentStepTimer->start();
    }
}

void AIAssistant::flushPendingAgentSteps(QNetworkReply *reply)
{
    if (!m_agentStepStreamers.contains(reply)) return;
    StreamingStepState st = m_agentStepStreamers.take(reply);
    QJsonArray &steps = m_agentStreamSteps[reply];
    if (!st.activeStep.isEmpty()) {
        steps.append(st.activeStep);
    }
    while (!st.pendingSteps.isEmpty()) {
        steps.append(st.pendingSteps.dequeue());
    }
    if (m_agentStepStreamers.isEmpty() && m_agentStepTimer && m_agentStepTimer->isActive()) {
        m_agentStepTimer->stop();
    }
}

void AIAssistant::onAgentStepTimerTick()
{
    if (m_agentStepStreamers.isEmpty()) {
        if (m_agentStepTimer && m_agentStepTimer->isActive()) m_agentStepTimer->stop();
        return;
    }
    const auto replies = m_agentStepStreamers.keys();
    for (QNetworkReply *reply : replies) {
        if (!m_agentStepStreamers.contains(reply)) continue;
        StreamingStepState &st = m_agentStepStreamers[reply];

        if (st.activeStep.isEmpty()) {
            if (st.pendingSteps.isEmpty()) {
                m_agentStepStreamers.remove(reply);
                continue;
            }
            st.activeStep = st.pendingSteps.dequeue();
            st.progress = 0;
            const QString type = st.activeStep.value("type").toString();
            if (type == "plan_reflection" || type == "verification") {
                st.targetLength = st.activeStep.value("result").toObject().value("reason").toString().length();
            } else if (type == "plan") {
                int total = 0;
                for (const QJsonValue &v : st.activeStep.value("steps").toArray()) {
                    total += v.toString().length();
                }
                st.targetLength = total;
            } else if (type == "thinking" || type == "step_answer") {
                st.targetLength = st.activeStep.value("content").toString().length();
            } else if (type == "tool_call") {
                const QString paramsStr = QString::fromUtf8(
                    QJsonDocument(st.activeStep.value("params").toObject()).toJson(QJsonDocument::Compact));
                st.targetLength = paramsStr.length();
            } else {
                st.targetLength = 0;
            }
        }

        const int speed = st.pendingSteps.isEmpty() ? 5 : (15 * (st.pendingSteps.size() + 1));
        st.progress += speed;

        const QString type = st.activeStep.value("type").toString();
        const bool stepDone = (st.targetLength <= 0 || st.progress >= st.targetLength);

        QJsonObject currentVisibleStep = st.activeStep;
        if (!stepDone) {
            currentVisibleStep["streaming"] = true;
            if (type == "plan_reflection" || type == "verification") {
                QJsonObject resObj = currentVisibleStep.value("result").toObject();
                const QString full = resObj.value("reason").toString();
                resObj["reason"] = full.left(st.progress);
                currentVisibleStep["result"] = resObj;
            } else if (type == "plan") {
                const QJsonArray fullArr = currentVisibleStep.value("steps").toArray();
                QJsonArray partialArr;
                int remaining = st.progress;
                for (const QJsonValue &v : fullArr) {
                    const QString s = v.toString();
                    if (remaining >= s.length()) {
                        partialArr.append(s);
                        remaining -= s.length();
                    } else if (remaining > 0) {
                        partialArr.append(s.left(remaining));
                        remaining = 0;
                    }
                }
                currentVisibleStep["steps"] = partialArr;
            } else if (type == "thinking" || type == "step_answer") {
                const QString full = currentVisibleStep.value("content").toString();
                currentVisibleStep["content"] = full.left(st.progress);
            }
        }

        QJsonArray visibleSteps = m_agentStreamSteps.value(reply);
        visibleSteps.append(currentVisibleStep);

        if (stepDone) {
            m_agentStreamSteps[reply].append(st.activeStep);
            st.activeStep = QJsonObject();
            st.progress = 0;
            st.targetLength = 0;
        }

        if (!m_agentStreamTexts.value(reply).isEmpty()) {
            QJsonObject draftAnswer;
            draftAnswer["type"] = QStringLiteral("final_answer");
            draftAnswer["content"] = m_agentStreamTexts.value(reply);
            draftAnswer["streaming"] = true;
            visibleSteps.append(draftAnswer);
        }

        QJsonObject envelope;
        envelope["status"] = QStringLiteral("running");
        envelope["stream_status"] = m_agentStreamStatuses.value(reply);
        envelope["steps"] = visibleSteps;
        updateAgentStreamDraft(reply, st.request, envelope);
    }
}

void AIAssistant::consumeCompletionStream(
    QNetworkReply *reply, const QueuedCompletionRequest &request) {
    QByteArray &buffer = m_streamBuffers[reply];
    buffer += reply->readAll();
    for (;;) {
        int boundary = buffer.indexOf("\n\n");
        int delimiterSize = 2;
        const int crlfBoundary = buffer.indexOf("\r\n\r\n");
        if (crlfBoundary >= 0 && (boundary < 0 || crlfBoundary < boundary)) {
            boundary = crlfBoundary;
            delimiterSize = 4;
        }
        if (boundary < 0) break;

        QByteArray event = buffer.left(boundary);
        buffer.remove(0, boundary + delimiterSize);
        event.replace("\r\n", "\n");
        const QList<QByteArray> lines = event.split('\n');
        QByteArray eventName;
        for (const QByteArray &line : lines) {
            if (line.startsWith("event:")) eventName = line.mid(6).trimmed();
        }
        for (const QByteArray &line : lines) {
            if (!line.startsWith("data:")) continue;
            const QByteArray data = line.mid(5).trimmed();
            if (data == "[DONE]") continue;
            const QJsonObject chunk = QJsonDocument::fromJson(data).object();
            if (request.isAgent) {
                if (eventName == "step") {
                    m_agentStreamTexts[reply].clear();
                    enqueueAgentStepForStreaming(reply, request, chunk);
                } else if (eventName == "status") {
                    const QString node = chunk.value("node").toString();
                    const QString phase = chunk.value("phase").toString();
                    const QString status = node.isEmpty()
                        ? chunk.value("status").toString()
                        : node + QStringLiteral(": ") + phase;
                    m_agentStreamStatuses.insert(reply, status);
                    QJsonObject envelope;
                    envelope["status"] = QStringLiteral("running");
                    envelope["stream_status"] = status;
                    envelope["steps"] = m_agentStreamSteps.value(reply);
                    updateAgentStreamDraft(reply, request, envelope);
                } else if (eventName == "delta") {
                    flushPendingAgentSteps(reply);
                    QString &answer = m_agentStreamTexts[reply];
                    answer += chunk.value("content").toString();
                    QJsonArray visibleSteps = m_agentStreamSteps.value(reply);
                    QJsonObject draftAnswer;
                    draftAnswer["type"] = QStringLiteral("final_answer");
                    draftAnswer["content"] = answer;
                    draftAnswer["streaming"] = true;
                    visibleSteps.append(draftAnswer);
                    QJsonObject envelope;
                    envelope["status"] = QStringLiteral("running");
                    envelope["stream_status"] = m_agentStreamStatuses.value(reply);
                    envelope["steps"] = visibleSteps;
                    updateAgentStreamDraft(reply, request, envelope);
                } else if (eventName == "done") {
                    flushPendingAgentSteps(reply);
                    m_agentStreamResults.insert(reply, chunk);
                    updateAgentStreamDraft(reply, request, chunk);
                } else if (eventName == "error") {
                    flushPendingAgentSteps(reply);
                    m_streamErrors.insert(
                        reply, chunk.value("detail").toString());
                }
                continue;
            }
            const QString error = chunk.value("error").toObject()
                                      .value("message").toString();
            if (!error.isEmpty()) {
                m_streamErrors.insert(reply, error);
                continue;
            }
            QString content = chunk.value("x_snapshot").toString();
            if (content.isNull()) {
                const QJsonArray choices = chunk.value("choices").toArray();
                if (!choices.isEmpty()) {
                    const QString delta = choices.first().toObject()
                                              .value("delta").toObject()
                                              .value("content").toString();
                    const int index = m_streamDraftIndices.value(reply, -1);
                    ChatSession *session = getSession(request.sessionId);
                    const QString current = session && index >= 0 && index < session->messages.size()
                        ? session->messages.at(index).content : QString();
                    content = current + delta;
                }
            }
            if (!content.isNull()) updateStreamDraft(reply, request, content);
        }
    }
}

void AIAssistant::onReplyReadyRead(QNetworkReply *reply) {
    if (!m_requests.hasReply(reply) || !isEventStream(reply)) return;
    consumeCompletionStream(reply, m_requests.requestForReply(reply));
}

void AIAssistant::clearStreamState(QNetworkReply *reply) {
    flushPendingAgentSteps(reply);
    m_streamBuffers.remove(reply);
    m_streamDraftIndices.remove(reply);
    m_streamErrors.remove(reply);
    m_agentStreamSteps.remove(reply);
    m_agentStreamResults.remove(reply);
    m_agentStreamTexts.remove(reply);
    m_agentStreamStatuses.remove(reply);
    m_agentStepStreamers.remove(reply);
}

void AIAssistant::onServerFinished(int exitCode, bool crashed) {
    m_serverReadyEmitted = false;

    if (hasPendingWork()) {
        const auto activeReplies = m_requests.pendingReplies();
        QList<QueuedCompletionRequest> activeRequests = m_requests.takeAllPending();

        for (QNetworkReply *reply : activeReplies) {
            if (reply) {
                clearStreamState(reply);
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
    
    const bool streamed = isEventStream(reply);
    if (streamed) consumeCompletionStream(reply, m_requests.requestForReply(reply));
    QueuedCompletionRequest request = m_requests.takeReply(reply);
    QString sessionId = request.sessionId;  // Remove from pending and get sessionId
    
    if (request.isAgent) {
        flushPendingAgentSteps(reply);
    }
    
    if (reply->error() == QNetworkReply::NoError) {
        if (request.isAgent) {
            flushPendingAgentSteps(reply);
            QJsonObject res = streamed
                ? m_agentStreamResults.value(reply)
                : QJsonDocument::fromJson(reply->readAll()).object();
            const QString streamError = m_streamErrors.value(reply);
            if (streamed && res.isEmpty()) {
                QJsonArray steps = m_agentStreamSteps.value(reply);
                if (!streamError.isEmpty()) {
                    QJsonObject errorStep;
                    errorStep["type"] = QStringLiteral("error");
                    errorStep["content"] = streamError;
                    steps.append(errorStep);
                }
                res["status"] = streamError.isEmpty()
                    ? QStringLiteral("incomplete")
                    : QStringLiteral("failed");
                res["steps"] = steps;
                updateAgentStreamDraft(reply, request, res);
            }

            // `application_action` is deliberately executed by the desktop app,
            // not by the Python server. The dispatcher only runs the action this
            // response names, never an earlier step included in its history.
            m_agentActions.dispatch(res, request.isUiActionAck,
                                    request.payload.value("request_id").toString(),
                                    sessionId, request.insertAfterIndex);
            
            // Append agent step message to history to keep it
            ChatMessage am;
            am.role = ChatRole::AgentTrace;
            am.content = QString::fromUtf8(QJsonDocument(res).toJson(QJsonDocument::Compact));
            am.timestamp = QDateTime::currentDateTime().toString(AppConstants::Format::chatTimestamp());
            
            ChatSession *sess = getSession(sessionId);
            if (sess) {
                const int draftIndex = m_streamDraftIndices.value(reply, -1);
                if (streamed && draftIndex >= 0 && draftIndex < sess->messages.size()) {
                    sess->messages[draftIndex] = am;
                } else if (request.insertAfterIndex >= 0 && request.insertAfterIndex < sess->messages.size()) {
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
        } else if (streamed) {
            const QString streamError = m_streamErrors.value(reply);
            const int draftIndex = m_streamDraftIndices.value(reply, -1);
            ChatSession *session = getSession(sessionId);
            if (!streamError.isEmpty()) {
                if (session && draftIndex >= 0 && draftIndex < session->messages.size()) {
                    session->messages[draftIndex].content +=
                        QStringLiteral("\n\n⚠ ") + streamError;
                } else {
                    appendAssistantMessage(sessionId, streamError, request.insertAfterIndex);
                }
            }
            saveAllSessions();
            emit responseReceived();
        } else {
            const QJsonObject response = QJsonDocument::fromJson(reply->readAll()).object();
            appendAssistantMessage(sessionId,
                                   AIAssistantPayloadBuilder::completionContent(response),
                                   request.insertAfterIndex);
            emit responseReceived();
        }
    } else {
        const QByteArray responseBody = reply->readAll();
        qWarning() << "[AIAssistant] Request failed:" << reply->errorString();

        if (isRecoverableConnectionError(reply->error()) && request.retryCount < 1 &&
            !streamed) {
            request.retryCount++;
            m_requests.prepend(request);

            clearStreamState(reply);
            reply->deleteLater();

            if (!m_server.isRunning()) {
                m_serverReadyEmitted = false;
                m_server.start(m_currentModelIndex);
            } else {
                processNextQueuedRequest();
            }

            m_isThinking = hasPendingWork();
            emit historyChanged();
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

        const int draftIndex = m_streamDraftIndices.value(reply, -1);
        ChatSession *session = getSession(sessionId);
        if (streamed && request.isAgent) {
            QJsonArray steps = m_agentStreamSteps.value(reply);
            QJsonObject errorStep;
            errorStep["type"] = QStringLiteral("error");
            errorStep["content"] = errorMessage;
            steps.append(errorStep);
            QJsonObject envelope;
            envelope["status"] = QStringLiteral("failed");
            envelope["steps"] = steps;
            updateAgentStreamDraft(reply, request, envelope);
            saveAllSessions();
        } else if (streamed && session && draftIndex >= 0 &&
            draftIndex < session->messages.size()) {
            session->messages[draftIndex].content +=
                QStringLiteral("\n\n⚠ ") + errorMessage;
            saveAllSessions();
        } else {
            appendAssistantMessage(sessionId, errorMessage, request.insertAfterIndex);
        }
    }
    
    processNextQueuedRequest();
    m_isThinking = hasPendingWork();
    emit historyChanged();
    
    clearStreamState(reply);
    reply->deleteLater();
}

// ── Persistence ───────────────────────────────────────────────────────────────

void AIAssistant::saveAllSessions() {
    m_sessions.save(m_auth ? m_auth->currentUsername() : QString(),
                    m_paths ? m_paths->configPath() : QString());
}

void AIAssistant::loadAllSessions() {
    m_sessions.load(m_auth ? m_auth->currentUsername() : QString(),
                    m_paths ? m_paths->configPath() : QString());
    // Older histories stored titleIsDefault=true even after the prompt had
    // replaced the title, so the sidebar kept rendering the placeholder.
    // Adopt the prompt once and persist the repair.
    if (m_sessions.adoptPromptTitles()) saveAllSessions();
}

void AIAssistant::reloadSessions() {
    loadAllSessions(); // selects the last stored session as current
    if (m_sessions.isEmpty()) {
        newChat();
    }
    emit sessionsChanged();
    emit historyChanged();
}
