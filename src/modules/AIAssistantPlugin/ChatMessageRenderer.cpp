#include "ChatMessageRenderer.h"
#include "ChatTemplates.h"
#include "../../utils/HtmlUtilities.h"
#include <QTextBrowser>
#include <QUrl>
#include <QFileInfo>
#include <QStringList>

namespace {
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
        QString preview = result.value("content").toString();
        if (!preview.isEmpty()) {
            const QStringList lines = preview.split('\n');
            preview = lines.mid(0, 12).join("\n");
            if (lines.size() > 12) preview += "\n…";
            html += "<pre style='margin:6px 0 0; padding:6px; background:#111827; color:#cbd5e1; max-height:180px; white-space:pre-wrap;'>" + preview.toHtmlEscaped() + "</pre>";
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

void ChatMessageRenderer::renderChatHistory(QTextBrowser* browser, IAppContext* ctx, const QList<QJsonObject>& history, bool isThinking, int thinkingInsertIndex) {
    if (!browser) return;
    
    QString h = ChatTemplates::CSS;
    h += QString("<div class='chat-start'>%1</div>").arg(ctx->translate("ai.new_chat_start"));
    
    QSet<QString> resolvedActions;
    for (const QJsonObject& message : history) {
        if (message["role"].toString() != "assistant_agent") continue;
        const QJsonArray steps = QJsonDocument::fromJson(message["content"].toString().toUtf8()).object()["steps"].toArray();
        for (const QJsonValue& value : steps) {
            const QJsonObject step = value.toObject();
            if (step["type"].toString() == "tool_result" && !step["action_id"].toString().isEmpty()) {
                resolvedActions.insert(step["action_id"].toString());
            }
        }
    }

    for (int i = 0; i < history.size(); ++i) {
        const auto &m = history[i];
        h += buildMessageHtml(m["role"].toString(), m["content"].toString(), m["attachments"].toArray(),
                              m["timestamp"].toString(), i, ctx, resolvedActions);
        
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
                                             const QSet<QString>& resolvedActions) {
    if (role == "assistant") {
        return ChatTemplates::AI_MESSAGE_CONTAINER.arg(HtmlUtilities::mdToHtml(content));
    } else if (role == "assistant_agent") {
        QJsonDocument doc = QJsonDocument::fromJson(content.toUtf8());
        QJsonObject res = doc.object();
        QJsonArray steps = res["steps"].toArray();
        const int priorStepCount = res["prior_step_count"].toInt();
        QString html;
        for (int i = priorStepCount; i < steps.size(); ++i) {
            QJsonObject step = steps[i].toObject();
            QString type = step["type"].toString();
            if (type == "thinking") {
                QString title = ctx->translate("ai.agent_thinking");
                if (title == "ai.agent_thinking") title = "🤔 " + ctx->translate("ai.thinking");
                if (title == "🤔 ai.thinking") title = "🤔 Thinking...";
                html += ChatTemplates::AGENT_THINKING.arg(title, step["content"].toString().toHtmlEscaped());
            } else if (type == "plan") {
                const QJsonArray planSteps = step["steps"].toArray();
                QString planHtml;
                for (int pi = 0; pi < planSteps.size(); ++pi) {
                    planHtml += QString("<div style='margin:2px 0;'><span style='color:#8b5cf6; font-weight:bold;'>%1.</span> %2</div>")
                        .arg(pi + 1)
                        .arg(planSteps[pi].toString().toHtmlEscaped());
                }
                html += ChatTemplates::AGENT_THINKING.arg("📌 Kế hoạch", planHtml);
            } else if (type == "delegation") {
                const QString agent = step["agent"].toString().toHtmlEscaped();
                const QString tool = step["tool"].toString().toHtmlEscaped();
                html += ChatTemplates::AGENT_THINKING.arg("Multi-Agent",
                    QString("Supervisor -> <b>%1</b> (%2)").arg(agent, tool));
            } else if (type == "tool_call") {
                QString toolName = step["tool"].toString();
                QString paramsStr = QString::fromUtf8(QJsonDocument(step["params"].toObject()).toJson(QJsonDocument::Compact));
                html += ChatTemplates::AGENT_TOOL_CALL.arg("🔧", toolName, paramsStr.toHtmlEscaped());
            } else if (type == "tool_result") {
                const QJsonObject result = step["result"].toObject();
                html += ChatTemplates::AGENT_TOOL_RESULT_LOCALIZED.arg(ctx->translate("ai.agent_result"),
                                                                         formatToolResult(step["tool"].toString(), result, ctx));
            } else if (type == "verification") {
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
                    innerAtt += ChatTemplates::IMAGE_ATTACHMENT.arg(path).arg(QUrl::fromLocalFile(path).toString());
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
