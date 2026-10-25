#include "ChatMessageRenderer.h"
#include "ChatTemplates.h"
#include "FileUtilities.h"
#include "../../utils/HtmlUtilities.h"
#include <QHash>
#include <QTextBrowser>
#include <QUrl>
#include <QFileInfo>
#include <QStringList>

namespace {

/// The chat history embeds the attachment image as a data: URL: the file on
/// disk is sealed, so the HTML renderer cannot read it from the file system
/// any more. The result is cached because the history is re-rendered on every
/// streamed answer.
QString attachmentImageDataUrl(const QString &path) {
    static QHash<QString, QString> cache;
    const auto cached = cache.constFind(path);
    if (cached != cache.constEnd()) return cached.value();

    QString thumbPath = path;
    thumbPath.replace(QStringLiteral("/Upload/"), QStringLiteral("/Thumbnails/"));
    QByteArray bytes = FileUtilities::readAttachment(thumbPath, QString::fromLatin1(FileUtilities::ThumbnailContext));
    if (bytes.isEmpty()) {
        bytes = FileUtilities::readAttachment(path, QString::fromLatin1(FileUtilities::AttachmentContext));
    }
    if (bytes.isEmpty()) return {};

    const bool isJpeg = path.endsWith(QStringLiteral(".jpg"), Qt::CaseInsensitive)
                        || path.endsWith(QStringLiteral(".jpeg"), Qt::CaseInsensitive);
    const QString mimeType = isJpeg ? QStringLiteral("image/jpeg") : QStringLiteral("image/png");
    const QString url = QStringLiteral("data:") + mimeType + QStringLiteral(";base64,")
                        + QString::fromLatin1(bytes.toBase64());
    if (cache.size() >= 24) cache.clear();
    cache.insert(path, url);
    return url;
}

int commonStepPrefixLength(const QJsonArray& previous, const QJsonArray& current) {
    const int limit = qMin(previous.size(), current.size());
    int index = 0;
    for (; index < limit; ++index) {
        if (QJsonDocument(previous[index].toObject()).toJson(QJsonDocument::Compact) !=
            QJsonDocument(current[index].toObject()).toJson(QJsonDocument::Compact)) {
            break;
        }
    }
    return index;
}

QString resultLine(const QString& label, const QString& value) {
    return QString("<div style='margin:2px 0;'><span style='color:#94a3b8;'>%1</span> <span style='color:#e2e8f0;'>%2</span></div>")
        .arg(label.toHtmlEscaped(), value.toHtmlEscaped());
}

QString formatToolResult(const QString& toolName, const QJsonObject& result, IAppContext* ctx) {
    if (result.contains("error")) {
        return QString("<span style='color:#fca5a5;'>%1</span>").arg(result.value("error").toString().toHtmlEscaped());
    }
    if (toolName == "get_project_status") {
        const QJsonArray changed = result.value("changed_files").toArray();
        QString html = resultLine(ctx->translate("ai.agent_branch"), result.value("git_branch").toString());
        html += resultLine(ctx->translate("ai.agent_changed_files"), QString::number(changed.size()));
        const QJsonObject counts = result.value("source_file_counts").toObject();
        QStringList countItems;
        for (auto it = counts.constBegin(); it != counts.constEnd(); ++it)
            countItems.append(QString("%1: %2").arg(it.key(), QString::number(it.value().toInt())));
        if (!countItems.isEmpty()) html += resultLine(ctx->translate("ai.agent_source_files"), countItems.join(" · "));
        if (!changed.isEmpty()) {
            html += "<div style='color:#94a3b8; margin-top:6px;'>" + ctx->translate("ai.agent_changed_preview") + "</div><ul style='margin:3px 0 0 15px; padding:0;'>";
            const int limit = qMin(10, changed.size());
            for (int i = 0; i < limit; ++i)
                html += "<li style='margin:1px 0; color:#cbd5e1;'>" + changed[i].toString().toHtmlEscaped() + "</li>";
            if (changed.size() > limit)
                html += "<li style='color:#94a3b8;'>" + ctx->translate("ai.agent_more_items").arg(changed.size() - limit) + "</li>";
            html += "</ul>";
        }
        return html;
    }
    if (toolName == "list_directory") {
        return resultLine(ctx->translate("ai.agent_directory"), result.value("path").toString()) +
               resultLine(ctx->translate("ai.agent_items"), QString::number(result.value("count").toInt()));
    }
    if (toolName == "search_text") {
        return resultLine(ctx->translate("ai.agent_matches"), QString::number(result.value("count").toInt())) +
               resultLine(ctx->translate("ai.agent_query"), result.value("query").toString());
    }
    if (toolName == "read_file") {
        QString html = resultLine(ctx->translate("ai.agent_file"), result.value("path").toString()) +
                       resultLine(ctx->translate("ai.agent_lines"), result.value("showing").toString());
        if (result.value("presentation").toString() == "citation_source") {
            return html + "<div style='color:#94a3b8; margin-top:6px;'>Full source is rendered in the styled citation below.</div>";
        }
        QString preview = result.value("content").toString();
        if (!preview.isEmpty()) {
            // The server now returns a bounded, symbol/line-scoped source
            // range. Render the complete range; the previous 12-line preview
            // silently hid the remainder of functions such as onRunReconstruction.
            html += "<pre style='margin:6px 0 0; padding:6px; background:#111827; color:#cbd5e1; white-space:pre-wrap;'>" + preview.toHtmlEscaped() + "</pre>";
        }
        return html;
    }
    if (result.value("success").toBool()) {
        return "<span style='color:#a7f3d0;'>" + ctx->translate("ai.agent_success") + "</span>" +
               (result.contains("path") ? resultLine(ctx->translate("ai.agent_file"), result.value("path").toString()) : QString());
    }
    return QString::fromUtf8(QJsonDocument(result).toJson(QJsonDocument::Compact)).toHtmlEscaped();
}
}

void ChatMessageRenderer::renderChatHistory(QTextBrowser* browser, IAppContext* ctx, const QList<ChatMessage>& history, bool isThinking, int thinkingInsertIndex) {
    if (!browser) return;
    
    QString h = ChatTemplates::CSS;
    h += QString("<div class='chat-start'>%1</div>").arg(ctx->translate("ai.new_chat_start"));
    
    QSet<QString> resolvedActions;
    for (const ChatMessage& message : history) {
        if (message.role != ChatRole::AgentTrace) continue;
        const QJsonArray steps = message.agentTrace()["steps"].toArray();
        for (const QJsonValue& value : steps) {
            const QJsonObject step = value.toObject();
            if (step["type"].toString() == "tool_result" && !step["action_id"].toString().isEmpty()) {
                resolvedActions.insert(step["action_id"].toString());
            }
        }
    }

    QJsonArray previousAgentSteps;
    for (int i = 0; i < history.size(); ++i) {
        const auto &m = history[i];
        int skipSteps = -1;
        if (m.role == ChatRole::AgentTrace) {
            const QJsonObject envelope = m.agentTrace();
            const QJsonArray currentSteps = envelope["steps"].toArray();
            // Server responses are snapshots.  Use the explicit cursor when
            // present and also derive the common prefix for older responses
            // that did not include prior_step_count.
            skipSteps = qMax(envelope["prior_step_count"].toInt(),
                             commonStepPrefixLength(previousAgentSteps, currentSteps));
            previousAgentSteps = currentSteps;
        } else {
            previousAgentSteps = QJsonArray();
        }
        QJsonArray attachments;
        for (const QString &attachment : m.attachments) attachments.append(attachment);
        h += buildMessageHtml(m.roleName(), m.content, attachments,
                              m.timestamp, i, ctx, resolvedActions, skipSteps);
        
        if (isThinking && thinkingInsertIndex == i) {
            h += QString("<div class='typing'>%1</div>").arg(ctx->translate("ai.thinking"));
        }
    }
    
    if (isThinking && (thinkingInsertIndex == -1 || thinkingInsertIndex >= history.size())) {
        h += QString("<div class='typing'>%1</div>").arg(ctx->translate("ai.thinking"));
    }
    
    browser->setHtml(h);
    browser->moveCursor(QTextCursor::End);
}

QString ChatMessageRenderer::buildMessageHtml(const QString &role, const QString &content, const QJsonArray &attachments,
                                             const QString &timestamp, int index, IAppContext* ctx,
                                             const QSet<QString>& resolvedActions, int skipSteps) {
    if (role == "assistant") {
        return ChatTemplates::AI_MESSAGE_CONTAINER.arg(HtmlUtilities::mdToHtml(content));
    } else if (role == "assistant_agent") {
        QJsonDocument doc = QJsonDocument::fromJson(content.toUtf8());
        QJsonObject res = doc.object();
        QJsonArray steps = res["steps"].toArray();
        const int priorStepCount = skipSteps >= 0 ? skipSteps : res["prior_step_count"].toInt();
        QString html;
        bool transferredToChatbot = false;
        for (const QJsonValue& value : steps) {
            const QJsonObject step = value.toObject();
            if (step["type"].toString() == "tool_call" &&
                step["tool"].toString() == "transfer_to_chatbot_agent") {
                transferredToChatbot = true;
                break;
            }
        }
        if (transferredToChatbot) {
            // A chatbot handoff is conversational output, not an agent-work
            // transcript.  Keep planning, delegation, tool and verification
            // details out of the Qt chat; render only the final response.
            for (const QJsonValue& value : steps) {
                const QJsonObject step = value.toObject();
                if (step["type"].toString() == "final_answer") {
                    html += ChatTemplates::AI_MESSAGE_CONTAINER.arg(
                        HtmlUtilities::mdToHtml(step["content"].toString()));
                }
            }
            return html;
        }
        // The planner can regenerate its plan several times while the plan
        // critic rejects earlier drafts.  Only the last (verified) plan is
        // shown so the chat does not repeat the same block once per attempt.
        int lastPlanIndex = -1;
        for (int i = priorStepCount; i < steps.size(); ++i) {
            if (steps[i].toObject()["type"].toString() == "plan") lastPlanIndex = i;
        }
        // Only the final verification outcome is shown; intermediate failed
        // drafts are folded away together with their plan blocks.
        int lastReflectionIndex = -1;
        for (int i = priorStepCount; i < steps.size(); ++i) {
            if (steps[i].toObject()["type"].toString() == "plan_reflection") lastReflectionIndex = i;
        }
        for (int i = priorStepCount; i < steps.size(); ++i) {
            QJsonObject step = steps[i].toObject();
            QString type = step["type"].toString();
            if (type == "thinking") {
                QString title = ctx->translate("ai.agent_thinking");
                if (title == "ai.agent_thinking") title = "🤔 " + ctx->translate("ai.thinking");
                if (title == "🤔 ai.thinking") title = "🤔 Thinking...";
                html += ChatTemplates::AGENT_THINKING.arg(title, step["content"].toString().toHtmlEscaped());
            } else if (type == "plan") {
                if (i != lastPlanIndex) continue; // superseded draft
                const QJsonArray planSteps = step["steps"].toArray();
                QString planHtml;
                for (int pi = 0; pi < planSteps.size(); ++pi) {
                    planHtml += QString("<div style='margin:2px 0;'><span style='color:#8b5cf6; font-weight:bold;'>%1.</span> %2</div>")
                        .arg(pi + 1)
                        .arg(planSteps[pi].toString().toHtmlEscaped());
                }
                html += ChatTemplates::AGENT_THINKING.arg(ctx->translate("ai.plan"), planHtml);
            } else if (type == "plan_reflection") {
                if (i != lastReflectionIndex) continue; // superseded draft
                const QJsonObject review = step["result"].toObject();
                const bool passed = review.value("passed").toBool();
                const QString status = passed ? "Plan verification passed" : "Plan verification failed";
                html += ChatTemplates::AGENT_THINKING.arg(status,
                    review.value("reason").toString().toHtmlEscaped());
            } else if (type == "delegation") {
                const QString agent = step["agent"].toString().toHtmlEscaped();
                const QString tool = step["tool"].toString().toHtmlEscaped();
                if (tool == "rag_search") continue;
                html += ChatTemplates::AGENT_THINKING.arg("Multi-Agent",
                    QString("Supervisor -> <b>%1</b> (%2)").arg(agent, tool));
            } else if (type == "tool_call") {
                QString toolName = step["tool"].toString();
                if (toolName == "rag_search") continue;
                QString paramsStr = QString::fromUtf8(QJsonDocument(step["params"].toObject()).toJson(QJsonDocument::Compact));
                html += ChatTemplates::AGENT_TOOL_CALL.arg("🔧", toolName, paramsStr.toHtmlEscaped());
            } else if (type == "tool_result") {
                if (step["tool"].toString() == "rag_search") continue;
                const QJsonObject result = step["result"].toObject();
                html += ChatTemplates::AGENT_TOOL_RESULT_LOCALIZED.arg(ctx->translate("ai.agent_result"),
                                                                         formatToolResult(step["tool"].toString(), result, ctx));
            } else if (type == "verification") {
                if (step["tool"].toString() == "rag_search") continue;
                const QJsonObject result = step["result"].toObject();
                const QString status = result.value("passed").toBool()
                    ? "Verification passed" : "Verification failed";
                html += ChatTemplates::AGENT_THINKING.arg(status,
                    result.value("reason").toString().toHtmlEscaped());
            } else if (type == "pending_approval") {
                QString toolName = step["tool"].toString();
                QString actionId = step["action_id"].toString();
                QString desc = step["description"].toString();
                const QJsonObject params = step["params"].toObject();
                if (toolName == "write_file") {
                    desc = ctx->translate("ai.agent_write_file").arg(params.value("path").toString());
                } else if (toolName == "patch_file") {
                    desc = ctx->translate("ai.agent_patch_file").arg(params.value("path").toString());
                } else if (toolName == "create_directory") {
                    desc = ctx->translate("ai.agent_create_directory").arg(params.value("path").toString());
                } else if (toolName == "run_command") {
                    desc = ctx->translate("ai.agent_run_command").arg(params.value("command").toString());
                }
                const bool resolved = resolvedActions.contains(actionId);
                if (resolved) {
                    html += ChatTemplates::AGENT_ACTION_PROCESSED.arg(ctx->translate("ai.agent_action_processed"),
                                                                         desc.toHtmlEscaped());
                } else {
                    html += ChatTemplates::AGENT_APPROVAL_BLOCK.arg(ctx->translate("ai.agent_approval_required").arg(toolName),
                                                                      desc.toHtmlEscaped(), actionId,
                                                                      ctx->translate("ai.agent_approve"),
                                                                      ctx->translate("ai.agent_reject"));
                }
            } else if (type == "final_answer") {
                html += ChatTemplates::AI_MESSAGE_CONTAINER.arg(HtmlUtilities::mdToHtml(step["content"].toString()));
            } else if (type == "error") {
                html += ChatTemplates::AI_MESSAGE_CONTAINER.arg("<span style='color:#ef4444;'>" + ctx->translate("ai.agent_error") + ": " + step["content"].toString().toHtmlEscaped() + "</span>");
            }
        }
        return html;
    } else {
        QString attHtml;
        if (!attachments.isEmpty()) {
            QString innerAtt;
            for (const auto &v : attachments) {
                QString path = v.toString();
                if (path.endsWith(".png") || path.endsWith(".jpg") || path.endsWith(".jpeg")) {
                    const QString imageUrl = attachmentImageDataUrl(path);
                    if (imageUrl.isEmpty()) {
                        // The attachment could not be unsealed: show its name
                        // instead of a broken image.
                        innerAtt += ChatTemplates::FILE_ATTACHMENT
                                    .arg(QUrl::fromLocalFile(path).toString())
                                    .arg(QFileInfo(path).fileName());
                    } else {
                        innerAtt += ChatTemplates::IMAGE_ATTACHMENT.arg(path).arg(imageUrl);
                    }
                } else if (FileUtilities::isSealedAttachment(path)) {
                    // A sealed file is ciphertext on disk: a "file://" link
                    // would hand the OS handler garbage, so the name is shown
                    // without a link.
                    innerAtt += ChatTemplates::FILE_ATTACHMENT_SEALED.arg(QFileInfo(path).fileName());
                } else {
                    innerAtt += ChatTemplates::FILE_ATTACHMENT
                                .arg(QUrl::fromLocalFile(path).toString())
                                .arg(QFileInfo(path).fileName());
                }
            }
            attHtml = ChatTemplates::ATTACHMENT_CONTAINER.arg(innerAtt);
        }

        QString textHtml;
        if (!content.isEmpty()) {
            QString escaped = content.toHtmlEscaped();
            escaped.replace("\n", "<br>");
            textHtml = ChatTemplates::USER_TEXT_TABLE.arg(escaped);
        }
        
        QString footerHtml = ChatTemplates::USER_ACTION_FOOTER;
        QDateTime dt = QDateTime::fromString(timestamp, "yyyy-MM-dd HH:mm:ss"); // Default chat format
        QString dateStr;
        if (dt.isValid()) {
            dateStr = dt.toString("MMM d");
        } else {
            dateStr = timestamp;
        }
        footerHtml = footerHtml.arg(dateStr).arg(index);

        return ChatTemplates::USER_MESSAGE_CONTAINER.arg(attHtml).arg(textHtml).arg(footerHtml);
    }
}
