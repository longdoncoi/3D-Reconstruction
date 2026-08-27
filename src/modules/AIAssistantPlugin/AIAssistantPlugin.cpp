#include "AIAssistantPlugin.h"
#include "MainWindow.h"
#include "IAppContext.h"
#include "ISettingsService.h"
#include "SignalBus.h"
#include "IAIAssistantService.h"
#include "LanguageManager.h"
#include "UserManager.h"
#include "AIAttachmentPreviewFactory.h"
#include "../../utils/FileUtilities.h"
#include <QDockWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QToolButton>
#include <QComboBox>
#include <QPushButton>
#include <QSplitter>
#include <QListWidget>
#include <QTextBrowser>
#include <QLineEdit>
#include <QFileDialog>
#include <QFileInfo>
#include <QInputDialog>
#include <QApplication>
#include <QDesktopServices>
#include <QMenu>
#include <QTimer>
#include <qclipboard.h>

#include "AIAssistantRibbonUI.h"
#include "ChatBotDockWidget.h"
#include "AppConfig.h"
#include "AppConstants.h"
#include "ChatMessageRenderer.h"
#include "../../utils/CustomProgressDialog.h"
#include "../../utils/ModernMessageBox.h"


void AIAssistantPlugin::initialize(IAppContext* context) {
    m_ctx = context;
    m_aiAssistant = m_ctx->services()->get<IAIAssistantService>();
    
    if (m_aiAssistant) {
        m_aiAssistant->reloadSessions();
    }
    
    setupChatbotUI();
    connect(m_dockUI->btnToggleAgentMode(), &QPushButton::toggled, this, [this](bool active) {
        m_dockUI->setAgentModeActive(active);
    });
    
    // Connect AIAssistant signals
    if (m_aiAssistant) {
        connect(m_aiAssistant, &IAIAssistantService::historyChanged,     this, &AIAssistantPlugin::updateChatUI);
        connect(m_aiAssistant, &IAIAssistantService::sessionsChanged,    this, &AIAssistantPlugin::updateSessionListUI);
        connect(m_aiAssistant, &IAIAssistantService::serverStatusChanged,this, &AIAssistantPlugin::onAssistantStatusChanged);
        connect(m_aiAssistant, &IAIAssistantService::errorOccurred,      this, &AIAssistantPlugin::onAssistantError);
        connect(m_aiAssistant, &IAIAssistantService::responseReceived,   this, &AIAssistantPlugin::updateChatUI);
        connect(m_aiAssistant, &IAIAssistantService::agentStepReceived,  this, [this](const QString &, const QJsonObject &) {
            m_submittedAgentActions.clear();
            updateChatUI();
        });
        connect(m_aiAssistant, &IAIAssistantService::agentApprovalRequired, this, [this](const QString &, const QJsonObject &){
            updateChatUI();
        });
        connect(m_aiAssistant, &IAIAssistantService::agentUiActionRequested, this,
                [this](const QString &action, const QVariantMap &parameters) {
            emit m_ctx->signalBus()->agentUiActionRequested(action, parameters);
        });
        connect(m_ctx->signalBus(), &SignalBus::agentUiActionCompleted, this,
                [this](const QString &requestId, bool success, const QVariantMap &result) {
            m_aiAssistant->reportUiActionResult(requestId, success, result);
        });
    }
    
    // Inject AI Assistant button into tab.ai_assistant panel
    if (QWidget* panel = m_ctx->getTabPanel("tab.ai_assistant")) {
        m_ribbonUI = new AIAssistantRibbonUI(m_ctx, panel, this);
        connect(m_ribbonUI->btnToggleAssistant(), &QToolButton::clicked, this, &AIAssistantPlugin::onToggleChatbot);
        connect(m_ribbonUI->btnRestartModel(), &QToolButton::clicked, this, [this]() {
            if (!m_aiAssistant) return;
            auto *btn = m_ribbonUI->btnRestartModel();
            btn->setEnabled(false);
            btn->setText(m_ctx->translate("ai.reloading"));
            if (m_progressDialog) {
                m_progressDialog->setLabelText(m_ctx->translate("ai.restarting_model"));
                m_progressDialog->setRange(0, 0);
                m_progressDialog->show();
                m_progressDialog->centerOnWidget(m_dockUI && !m_dockUI->dockWidget()->isHidden() ? static_cast<QWidget*>(m_dockUI->dockWidget()) : static_cast<QWidget*>(m_ctx->mainWindow()));
            }
            m_aiAssistant->restartModel();
            QTimer::singleShot(5000, btn, [this, btn]() {
                btn->setEnabled(true);
                btn->setText(m_ctx->translate("ai.restart_model"));
            });
        });
        connect(m_ribbonUI->btnRestartRAG(), &QToolButton::clicked, this, [this]() {
            if (!m_aiAssistant) return;
            auto *btn = m_ribbonUI->btnRestartRAG();
            btn->setEnabled(false);
            btn->setText(m_ctx->translate("ai.reloading"));
            if (m_progressDialog) {
                m_progressDialog->setLabelText(m_ctx->translate("ai.restarting_rag"));
                m_progressDialog->setRange(0, 0);
                m_progressDialog->show();
                m_progressDialog->centerOnWidget(m_dockUI && !m_dockUI->dockWidget()->isHidden() ? static_cast<QWidget*>(m_dockUI->dockWidget()) : static_cast<QWidget*>(m_ctx->mainWindow()));
            }
            m_aiAssistant->restartRAG();
            QTimer::singleShot(5000, btn, [this, btn]() {
                btn->setEnabled(true);
                btn->setText(m_ctx->translate("ai.restart_rag"));
            });
        });
        connect(m_ribbonUI->btnRestartAgent(), &QToolButton::clicked, this, [this]() {
            if (!m_aiAssistant) return;
            if (m_progressDialog) {
                m_progressDialog->setLabelText(m_ctx->translate("ai.restarting_agent"));
                m_progressDialog->setRange(0, 0);
                m_progressDialog->show();
                m_progressDialog->centerOnWidget(m_dockUI && !m_dockUI->dockWidget()->isHidden() ? static_cast<QWidget*>(m_dockUI->dockWidget()) : static_cast<QWidget*>(m_ctx->mainWindow()));
            }
            m_aiAssistant->restartAgent();
        });
        connect(m_ribbonUI->btnRestartServer(), &QToolButton::clicked, this, [this]() {
            if (!m_aiAssistant) return;
            auto *btn = m_ribbonUI->btnRestartServer();
            btn->setEnabled(false);
            btn->setText(m_ctx->translate("ai.reloading"));
            if (m_progressDialog) {
                m_progressDialog->setLabelText(m_ctx->translate("ai.restarting_server"));
                m_progressDialog->setRange(0, 0);
                m_progressDialog->show();
                m_progressDialog->centerOnWidget(m_dockUI && !m_dockUI->dockWidget()->isHidden() ? static_cast<QWidget*>(m_dockUI->dockWidget()) : static_cast<QWidget*>(m_ctx->mainWindow()));
            }
            m_aiAssistant->restartServer();
            QTimer::singleShot(5000, btn, [this, btn]() {
                btn->setEnabled(true);
                btn->setText(m_ctx->translate("ai.restart_server"));
            });
        });
    }
    
    // Connect SignalBus for retranslation
    connect(m_ctx->signalBus(), &SignalBus::languageChanged, this, [this](const QString &) {
        if (m_dockUI) {
            m_dockUI->dockWidget()->setWindowTitle(m_ctx->translate("ai.dock_title"));
            m_dockUI->btnNewChat()->setText(m_ctx->translate("ai.new_chat"));
            m_dockUI->chatInput()->setPlaceholderText(m_ctx->translate("ai.input_hint"));
            m_dockUI->sessLabel()->setText(m_ctx->translate("ai.recent"));

            m_dockUI->btnAttach()->setToolTip(m_ctx->translate("ai.attach"));
            m_dockUI->actAttachImage()->setText(m_ctx->translate("ai.attach_image"));
            m_dockUI->actAttachFile()->setText(m_ctx->translate("ai.attach_file"));
            m_dockUI->setAgentModeActive(m_dockUI->btnToggleAgentMode()->isChecked());
        }

        // Update Ribbon UI button and Groupbox
        if (m_ribbonUI) {
            bool visible = m_dockUI && m_dockUI->dockWidget()->isVisible();
            m_ribbonUI->btnToggleAssistant()->setText(visible ? m_ctx->translate("ai.close_assistant") : m_ctx->translate("ai.open_assistant"));
            m_ribbonUI->btnRestartModel()->setText(m_ctx->translate("ai.restart_model"));
            m_ribbonUI->btnRestartRAG()->setText(m_ctx->translate("ai.restart_rag"));
            m_ribbonUI->btnRestartAgent()->setText(m_ctx->translate("ai.restart_agent"));
            m_ribbonUI->btnRestartServer()->setText(m_ctx->translate("ai.restart_server"));
            if (QLabel *lbl = m_ribbonUI->groupAI()->findChild<QLabel*>("groupTitleLabel")) {
                lbl->setText(m_ctx->translate("menu.ai_assistant"));
            }
        }
        updateSessionListUI();
        updateChatUI();
    });
    connect(m_ctx->signalBus(), &SignalBus::agentUiActionRequested, this,
            [this](const QString &action, const QVariantMap &parameters) {
        if (!m_dockUI) {
            const QString requestId = parameters.value("request_id").toString();
            if (!requestId.isEmpty() && action.startsWith("assistant."))
                emit m_ctx->signalBus()->agentUiActionCompleted(requestId, false, QVariantMap{{"error", "Assistant UI is unavailable"}});
            return;
        }
        bool handled = false;
        if (action == "assistant.open" && m_dockUI->dockWidget()->isHidden()) {
            onToggleChatbot();
            handled = true;
        } else if (action == "assistant.close" && !m_dockUI->dockWidget()->isHidden()) {
            onToggleChatbot();
            handled = true;
        } else if (action == "assistant.reload_model") {
            m_aiAssistant->restartModel();
            handled = true;
        } else if (action == "assistant.reload_rag") {
            m_aiAssistant->restartRAG();
            handled = true;
        } else if (action == "assistant.reload_agent") {
            m_aiAssistant->restartAgent();
            handled = true;
        } else if (action == "assistant.reload_server") {
            m_aiAssistant->restartServer();
            handled = true;
        }
        const QString requestId = parameters.value("request_id").toString();
        if (!requestId.isEmpty() && action.startsWith("assistant."))
            emit m_ctx->signalBus()->agentUiActionCompleted(requestId, handled,
                QVariantMap{{"action", action}, {"error", handled ? "" : "Action was not handled"}});
    });
    
    m_progressDialog = new CustomProgressDialog(m_ctx->mainWindow());
    connect(m_progressDialog, &CustomProgressDialog::stopRequested, this, &AIAssistantPlugin::onProgressStopped);

    connect(m_ctx->signalBus(), &SignalBus::userChanged, this, [this](const QString &username) {
        if (m_aiAssistant) {
            m_aiAssistant->reloadSessions();
            // Sessions are loaded synchronously; update UI right after
            updateSessionListUI();
            updateChatUI();
        }
        if (m_dockUI && m_dockUI->modelSelector()) {
            int savedIdx = 0;
            if (auto *um = UserManager::instance()) {
                savedIdx = um->getUserPref(username, "ai_model_index", "0").toInt();
            }
            m_dockUI->modelSelector()->blockSignals(true);
            m_dockUI->modelSelector()->setCurrentIndex(savedIdx);
            m_dockUI->modelSelector()->blockSignals(false);
        }
    });
}

void AIAssistantPlugin::cleanup() {
}

void AIAssistantPlugin::setupChatbotUI() {
    m_dockUI = new ChatBotDockWidget(m_ctx, this);

    connect(m_dockUI->dockWidget(), &QDockWidget::visibilityChanged, this, [this](bool visible) {
        if (m_ribbonUI) {
            m_ribbonUI->btnToggleAssistant()->setText(visible ? m_ctx->translate("ai.close_assistant") : m_ctx->translate("ai.open_assistant"));
        }
    });

    connect(m_dockUI->btnSendChat(), &QPushButton::clicked,    this, &AIAssistantPlugin::onSendChatMessage);
    connect(m_dockUI->chatInput(),   &QLineEdit::returnPressed, this, &AIAssistantPlugin::onSendChatMessage);
    connect(m_dockUI->btnToggleHistory(), &QToolButton::clicked, this, &AIAssistantPlugin::onToggleSessionHistory);
    connect(m_dockUI->modelSelector(), QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AIAssistantPlugin::onModelSelected);
    connect(m_dockUI->btnNewChat(), &QPushButton::clicked, this, &AIAssistantPlugin::onNewChat);
    connect(m_dockUI->sessionListWidget(), &QListWidget::itemClicked, this, &AIAssistantPlugin::onSessionClicked);
    connect(m_dockUI->sessionListWidget(), &QListWidget::customContextMenuRequested, this, &AIAssistantPlugin::onSessionMenuRequested);
    connect(m_dockUI->chatHistory(), &QTextBrowser::anchorClicked, this, &AIAssistantPlugin::onChatLinkClicked);
    connect(m_dockUI->actAttachImage(), &QAction::triggered, this, &AIAssistantPlugin::onAttachImage);
    connect(m_dockUI->actAttachFile(), &QAction::triggered, this, &AIAssistantPlugin::onAttachFile);
    
    updateSessionListUI();
    updateChatUI();
}

void AIAssistantPlugin::onToggleChatbot() {
    if (!m_dockUI) return;
    if (m_dockUI->dockWidget()->isHidden()) {
        m_dockUI->dockWidget()->show();
        if (!m_aiAssistant->isServerRunning()) {
            m_isStartingServer = true;
            if (m_progressDialog) {
                m_progressDialog->setLabelText(m_ctx->translate("ai.starting_server"));
                m_progressDialog->setRange(0, 0);
                m_progressDialog->show();
                m_progressDialog->centerOnWidget(m_dockUI->dockWidget());
            }
            m_aiAssistant->startServer(m_dockUI->modelSelector()->currentIndex());
        }
    } else {
        m_dockUI->dockWidget()->hide();
    }
}

void AIAssistantPlugin::onNewChat() {
    // Create new session immediately
    m_aiAssistant->newChat();
    updateSessionListUI();
    updateChatUI();
    if (m_dockUI) {
        m_dockUI->chatInput()->clear();
        m_dockUI->chatInput()->setFocus();  // Focus input for next message
    }
}

void AIAssistantPlugin::onToggleSessionHistory() {
    if (!m_dockUI) return;
    QSplitter *splitter = m_dockUI->chatSplitter();
    if (!splitter) return;

    m_dockUI->sessionPanel()->show();
    const QList<int> sizes = splitter->sizes();
    const int currentSessionWidth = sizes.value(0);
    const int currentChatWidth = sizes.value(1);
    const int totalWidth = qMax(currentSessionWidth + currentChatWidth, splitter->width());

    if (currentSessionWidth > 0) {
        splitter->setSizes({0, totalWidth});
        return;
    }

    const int sessionWidth = qBound(AppConstants::AIAssistant::SESSION_PANEL_MIN_WIDTH,
                                    totalWidth / 3,
                                    AppConstants::AIAssistant::SESSION_PANEL_MAX_WIDTH);
    splitter->setSizes({sessionWidth, qMax(0, totalWidth - sessionWidth)});
}

void AIAssistantPlugin::onSendChatMessage() {
    if (!m_dockUI) return;
    
    QString tx = m_dockUI->chatInput()->text().trimmed(); 
    if (tx.isEmpty() && pendingAttachments.isEmpty()) return; 
    m_dockUI->chatInput()->clear(); 
    
    QStringList atts = pendingAttachments;
    pendingAttachments.clear();
    QLayoutItem *child;
    while ((child = m_dockUI->attachmentLayout()->takeAt(0)) != nullptr) {
        delete child->widget();
        delete child;
    }
    m_dockUI->attachmentPreviewArea()->hide();
    constexpr bool isAgentMode = true;
    
    qDebug() << "[AIAssistantPlugin] Mode đang sử dụng:" << (isAgentMode ? "Agent Model" : "Chat Model") << "| Nội dung:" << tx;

    // Get selected sessions (allow multi-select)
    QList<QListWidgetItem*> selectedItems = m_dockUI->sessionListWidget()->selectedItems();
    if (selectedItems.isEmpty()) {
        // If no selection, send to current session
        m_aiAssistant->executeAgentTask(m_aiAssistant->currentSessionId(), tx, atts);
    } else if (selectedItems.size() == 1) {
        // Single selection - send to that session
        QString sessionId = selectedItems[0]->data(Qt::UserRole).toString();
        m_aiAssistant->executeAgentTask(sessionId, tx, atts);
    } else {
        // Multiple selections - send the same message to all selected sessions
        for (QListWidgetItem* item : selectedItems) {
            QString sessionId = item->data(Qt::UserRole).toString();
            m_aiAssistant->executeAgentTask(sessionId, tx, atts);
        }
    }
    
    m_dockUI->btnSendChat()->setEnabled(false); 
}

#include "ChatImageViewer.h"

void AIAssistantPlugin::onModelSelected(int index) {
    m_isStartingServer = true;
    if (m_progressDialog) {
        m_progressDialog->setLabelText(m_ctx->translate("ai.starting_server"));
        m_progressDialog->setRange(0, 0);
        m_progressDialog->show();
        if (m_dockUI) m_progressDialog->centerOnWidget(m_dockUI->dockWidget());
    }

    m_aiAssistant->switchModel(index);

    // Save per-user AI model index
    if (auto *um = UserManager::instance()) {
        um->setUserPref(um->currentUsername(), "ai_model_index", QString::number(index));
    }
}

void AIAssistantPlugin::onChatLinkClicked(const QUrl &url) {
    if (url.scheme() == "action") {
        QString path = url.path();
        if (path.startsWith("retry:")) {
            int msgIndex = path.mid(6).toInt();
            m_aiAssistant->retryAgentTask(m_aiAssistant->currentSessionId(), msgIndex);
        } else if (path.startsWith("edit:")) {
            int msgIndex = path.mid(5).toInt();
            auto history = m_aiAssistant->getHistory();
            if (msgIndex >= 0 && msgIndex < history.size()) {
                QString currentText = history[msgIndex]["content"].toString();
                bool ok;
                QString newText = QInputDialog::getMultiLineText(m_ctx->mainWindow(), 
                                      "Edit Message", 
                                      "Update your question:", 
                                      currentText, &ok);
                if (ok && !newText.isEmpty() && newText != currentText) {
                    m_aiAssistant->editMessage(m_aiAssistant->currentSessionId(), msgIndex, newText);
                }
            }
        }
        return;
    }

    if (url.scheme() == "agent") {
        QString path = url.path();
        if (path.startsWith("approve:")) {
            QString actionId = path.mid(8);
            if (m_submittedAgentActions.contains(actionId)) return;
            if (!ModernMessageBox::question(m_ctx->mainWindow(), m_ctx->translate("ai.agent_approve_title"),
                                            m_ctx->translate("ai.agent_approve_confirm"))) return;
            m_submittedAgentActions.insert(actionId);
            m_aiAssistant->approveAgentAction(m_aiAssistant->currentSessionId(), actionId);
        } else if (path.startsWith("reject:")) {
            QString actionId = path.mid(7);
            if (m_submittedAgentActions.contains(actionId)) return;
            if (!ModernMessageBox::question(m_ctx->mainWindow(), m_ctx->translate("ai.agent_reject_title"),
                                            m_ctx->translate("ai.agent_reject_confirm"))) return;
            m_submittedAgentActions.insert(actionId);
            m_aiAssistant->rejectAgentAction(m_aiAssistant->currentSessionId(), actionId);
        }
        return;
    }

    if (url.scheme() == "img") {
        QString path = url.path();
#ifdef Q_OS_WIN
        if (path.startsWith("/")) path.remove(0, 1);
#endif
        ChatImageViewer viewer(path, m_ctx->mainWindow());
        viewer.exec();
    } else if (url.scheme() == "file") {
        QString path = url.path();
        // Remove leading slash if it exists (for file:///path)
        if (path.startsWith("/")) path.remove(0, 1);
        
        // Resolve relative path against the configured project root.
        QString absPath = QDir(AppConfig::instance().projectRootDir()).absoluteFilePath(path);
        
        if (QFileInfo::exists(absPath)) {
            QDesktopServices::openUrl(QUrl::fromLocalFile(absPath));
        } else {
            QDesktopServices::openUrl(url);
        }
    } else {
        QDesktopServices::openUrl(url);
    }
}

void AIAssistantPlugin::updateSessionListUI() {
    if (!m_dockUI) return;
    m_dockUI->sessionListWidget()->clear();
    auto sessions = m_aiAssistant->getSessions();
    QString currentId = m_aiAssistant->currentSessionId();
    for (const auto &s : sessions) {
        QString displayTitle = s.title;
        if (displayTitle.startsWith("Phiên mới ") || displayTitle.startsWith("New Session ")) {
            QString dateTimeStr;
            if (displayTitle.startsWith("Phiên mới ")) {
                dateTimeStr = displayTitle.mid(10).trimmed();
            } else {
                dateTimeStr = displayTitle.mid(12).trimmed();
            }
            
            // Recursively clean up any legacy artifacts from saved sessions (e.g. "mới " or "Session ")
            if (dateTimeStr.startsWith("mới ")) {
                dateTimeStr = dateTimeStr.mid(4).trimmed();
            }
            if (dateTimeStr.startsWith("Session ")) {
                dateTimeStr = dateTimeStr.mid(8).trimmed();
            }
            if (dateTimeStr.startsWith("mới ")) {
                dateTimeStr = dateTimeStr.mid(4).trimmed();
            }
            
            displayTitle = m_ctx->translate("ai.new_session") + " " + dateTimeStr;
        }
        QListWidgetItem *item = new QListWidgetItem(displayTitle, m_dockUI->sessionListWidget());
        item->setData(Qt::UserRole, s.id);
        if (s.id == currentId) m_dockUI->sessionListWidget()->setCurrentItem(item);
    }
}

void AIAssistantPlugin::onSessionClicked(QListWidgetItem *item) {
    if (!item) return;
    m_aiAssistant->loadSession(item->data(Qt::UserRole).toString());
    updateChatUI();
}

void AIAssistantPlugin::onSessionMenuRequested(const QPoint &pos) {
    if (!m_dockUI) return;
    QListWidgetItem *item = m_dockUI->sessionListWidget()->itemAt(pos);
    if (!item) return;
    QString sid = item->data(Qt::UserRole).toString();
    QMenu menu;
    menu.addAction(m_ctx->translate("ai.copy_chat"), [this, sid]() {
        auto sessions = m_aiAssistant->getSessions();
        for (const auto &s : sessions) {
            if (s.id == sid) {
                QString fullChat;
                for (const auto &msg : s.messages) {
                    QString role = msg["role"].toString() == "user" ? "User: " : "AI: ";
                    fullChat += role + msg["content"].toString() + "\n\n";
                }
                QApplication::clipboard()->setText(fullChat);
                break;
            }
        }
    });
    menu.addSeparator();
    menu.addAction(m_ctx->translate("ai.delete_chat"),
                   [this, sid]() { m_aiAssistant->deleteSession(sid); });
    menu.exec(m_dockUI->sessionListWidget()->mapToGlobal(pos));
}

void AIAssistantPlugin::onAssistantStatusChanged(const QString &status) {
    if (!m_dockUI) return;
    if (status == m_ctx->translate("ai.starting_server")) return;
    
    if (status == m_ctx->translate("ai.server_ready") || status.startsWith(QString::fromUtf8("✅ "))) {
        m_isStartingServer = false;
        if (m_progressDialog) m_progressDialog->hide();
        m_dockUI->chatHistory()->append("<font color='#00A36C'><b>" + status + "</b></font>");
    } else {
        if (m_isStartingServer && m_progressDialog) {
            m_progressDialog->setLabelText(status);
        } else {
            m_dockUI->chatHistory()->append("<i>" + status + "</i>");
        }
    }
    m_dockUI->chatHistory()->moveCursor(QTextCursor::End);
}

void AIAssistantPlugin::onAssistantError(const QString &error) {
    if (!m_dockUI) return;
    m_dockUI->chatHistory()->append("<font color='red'>" + error + "</font>");
    m_dockUI->chatHistory()->moveCursor(QTextCursor::End);
    m_isStartingServer = false;
    if (m_progressDialog) m_progressDialog->hide();
}

void AIAssistantPlugin::onAttachImage() {
    QString lastUsedPath = m_ctx->settings()->getLastUsedPath("ai_attach");
    QStringList fileNames = QFileDialog::getOpenFileNames(m_ctx->mainWindow(), m_ctx->translate("file.select_image"), lastUsedPath, "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fileNames.isEmpty()) return;
    
    m_ctx->settings()->setLastUsedPath("ai_attach", QFileInfo(fileNames.first()).absolutePath());
    
    for (const QString &fileName : fileNames) {
        if (QFileInfo(fileName).size() > AppConstants::AIAssistant::MAX_ATTACHMENT_SIZE_MB * 1024 * 1024) {
            ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("ai.over_size"), 
                m_ctx->translate("ai.over_size_msg")
                .arg(QFileInfo(fileName).fileName())
                .arg(AppConstants::AIAssistant::MAX_ATTACHMENT_SIZE_MB));
            continue;
        }
        addAttachmentPreview(fileName, true);
    }
}

void AIAssistantPlugin::onAttachFile() {
    QString lastUsedPath = m_ctx->settings()->getLastUsedPath("ai_attach");
    QStringList fileNames = QFileDialog::getOpenFileNames(m_ctx->mainWindow(), m_ctx->translate("file.select_file"), lastUsedPath, "All Files (*.*)");
    if (fileNames.isEmpty()) return;
    
    m_ctx->settings()->setLastUsedPath("ai_attach", QFileInfo(fileNames.first()).absolutePath());
    
    for (const QString &fileName : fileNames) {
        if (QFileInfo(fileName).size() > AppConstants::AIAssistant::MAX_ATTACHMENT_SIZE_MB * 1024 * 1024) { 
            ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("ai.over_size"), 
                m_ctx->translate("ai.over_size_msg")
                .arg(QFileInfo(fileName).fileName())
                .arg(AppConstants::AIAssistant::MAX_ATTACHMENT_SIZE_MB));
            continue;
        }
        addAttachmentPreview(fileName, false);
    }
}

void AIAssistantPlugin::addAttachmentPreview(const QString &filePath, bool isImage) {
    if (!m_dockUI) return;
    FileUtilities::AttachmentResult result = FileUtilities::processAttachment(
        filePath,
        isImage,
        AppConfig::instance().uploadDir(),
        AppConfig::instance().thumbnailsDir());

    if (!result.success) return;

    pendingAttachments.append(result.destPath);

    QWidget *previewWidget = AIAttachmentPreviewFactory::create(
        m_dockUI->attachmentPreviewArea(),
        result,
        [this, result](QWidget *widget) {
            removeAttachment(result.destPath, widget);
        });

    m_dockUI->attachmentLayout()->addWidget(previewWidget);
    m_dockUI->attachmentPreviewArea()->show();
}

void AIAssistantPlugin::removeAttachment(const QString &filePath, QWidget *previewWidget) {
    if (!m_dockUI) return;
    // Delete both the original file and its thumbnail
    FileUtilities::deleteAttachment(filePath);
    
    // Construct thumbnail path from original path
    QString thumbPath = filePath;
    thumbPath.replace("/Upload/", "/Thumbnails/");
    if (!filePath.endsWith(".png") && !filePath.endsWith(".jpg") && !filePath.endsWith(".jpeg")) {
        thumbPath += ".png";
    }
    FileUtilities::deleteAttachment(thumbPath);

    pendingAttachments.removeOne(filePath);
    m_dockUI->attachmentLayout()->removeWidget(previewWidget);
    previewWidget->deleteLater();
    if (pendingAttachments.isEmpty()) {
        m_dockUI->attachmentPreviewArea()->hide();
    }
}

void AIAssistantPlugin::updateChatUI() {
    if (!m_dockUI || !m_dockUI->chatHistory()) return;
    
    // Disable send button if current session is thinking
    bool currentSessionThinking = m_aiAssistant->isSessionThinking(m_aiAssistant->currentSessionId());
    m_dockUI->btnSendChat()->setEnabled(!currentSessionThinking);
    
    // Always allow "New Chat" even when thinking
    m_dockUI->btnNewChat()->setEnabled(true);
    
    // Disable input if current session is thinking
    m_dockUI->chatInput()->setEnabled(!currentSessionThinking);
    
    m_dockUI->chatHistory()->clear(); 
    
    auto history = m_aiAssistant->getHistory();
    const int thinkingInsertIndex = m_aiAssistant->sessionThinkingInsertIndex(m_aiAssistant->currentSessionId());
    ChatMessageRenderer::renderChatHistory(m_dockUI->chatHistory(), m_ctx, history, currentSessionThinking, thinkingInsertIndex);
}

void AIAssistantPlugin::onProgressStopped() {
    if (m_isStartingServer) {
        m_aiAssistant->stopServer();
        m_isStartingServer = false;
        if (m_progressDialog) m_progressDialog->hide();
        if (m_dockUI) {
            m_dockUI->chatHistory()->append("<font color='red'>" + m_ctx->translate("ai.cancel_server") + "</font>");
            m_dockUI->chatHistory()->moveCursor(QTextCursor::End);
        }
    }
}
