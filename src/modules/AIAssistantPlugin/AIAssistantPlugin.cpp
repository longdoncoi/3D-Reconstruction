#include "AIAssistantPlugin.h"
#include "MainWindow.h"
#include "IAppContext.h"
#include "ISettingsService.h"
#include "SignalBus.h"
#include "IAIAssistantService.h"
#include "LanguageManager.h"
#include "UserManager.h"
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
#include <QMessageBox>
#include <QInputDialog>
#include <QSettings>
#include <QApplication>
#include <QDesktopServices>
#include <QJsonArray>
#include <QRegularExpression>
#include <QMenu>
#include <QMenuBar>
#include <qclipboard.h>
#include <QPainter>
#include <QLinearGradient>

#include "AIAssistantRibbonUI.h"
#include "ChatBotDockWidget.h"
#include "AppConfig.h"
#include "AppConstants.h"
#include "ChatTemplates.h"
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
    
    // Connect AIAssistant signals
    if (m_aiAssistant) {
        connect(m_aiAssistant, &IAIAssistantService::historyChanged,     this, &AIAssistantPlugin::updateChatUI);
        connect(m_aiAssistant, &IAIAssistantService::sessionsChanged,    this, &AIAssistantPlugin::updateSessionListUI);
        connect(m_aiAssistant, &IAIAssistantService::serverStatusChanged,this, &AIAssistantPlugin::onAssistantStatusChanged);
        connect(m_aiAssistant, &IAIAssistantService::errorOccurred,      this, &AIAssistantPlugin::onAssistantError);
        connect(m_aiAssistant, &IAIAssistantService::responseReceived,   this, &AIAssistantPlugin::updateChatUI);
    }
    
    // Inject AI Assistant button into tab.ai_assistant panel
    if (QWidget* panel = m_ctx->getTabPanel("tab.ai_assistant")) {
        m_ribbonUI = new AIAssistantRibbonUI(m_ctx, panel, this);
        connect(m_ribbonUI->btnToggleAssistant(), &QToolButton::clicked, this, &AIAssistantPlugin::onToggleChatbot);
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
        }

        // Update Ribbon UI button and Groupbox
        if (m_ribbonUI) {
            bool visible = m_dockUI && m_dockUI->dockWidget()->isVisible();
            m_ribbonUI->btnToggleAssistant()->setText(visible ? m_ctx->translate("ai.close_assistant") : m_ctx->translate("ai.open_assistant"));
            if (QLabel *lbl = m_ribbonUI->groupAI()->findChild<QLabel*>("groupTitleLabel")) {
                lbl->setText(m_ctx->translate("menu.ai_assistant"));
            }
        }
        updateSessionListUI();
        updateChatUI();
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
    
    // Get selected sessions (allow multi-select)
    QList<QListWidgetItem*> selectedItems = m_dockUI->sessionListWidget()->selectedItems();
    if (selectedItems.isEmpty()) {
        // If no selection, send to current session
        m_aiAssistant->sendMessage(tx, atts);
    } else if (selectedItems.size() == 1) {
        // Single selection - send to that session
        QString sessionId = selectedItems[0]->data(Qt::UserRole).toString();
        m_aiAssistant->sendMessageToSession(sessionId, tx, atts);
    } else {
        // Multiple selections - send the same message to all selected sessions
        for (QListWidgetItem* item : selectedItems) {
            QString sessionId = item->data(Qt::UserRole).toString();
            m_aiAssistant->sendMessageToSession(sessionId, tx, atts);
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
            m_aiAssistant->retryMessage(m_aiAssistant->currentSessionId(), msgIndex);
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
        
        // Resolve relative path against project root
        QString projectRoot = QDir::cleanPath(QApplication::applicationDirPath() + "/../../");
        QString absPath = QDir(projectRoot).absoluteFilePath(path);
        
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
    
    if (status == m_ctx->translate("ai.server_ready")) {
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
    if (m_isStartingServer) {
        m_isStartingServer = false;
        if (m_progressDialog) m_progressDialog->hide();
    }
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
    QString projectRoot = QDir::cleanPath(QApplication::applicationDirPath() + "/../../");
    FileUtilities::AttachmentResult result = FileUtilities::processAttachment(filePath, isImage, projectRoot);

    if (!result.success) return;

    pendingAttachments.append(result.destPath);
    // We don't append thumbPath to pendingAttachments to avoid double display in chat history.

    QWidget *previewWidget = new QWidget(m_dockUI->attachmentPreviewArea());
    previewWidget->setFixedSize(84, 84);
    previewWidget->setStyleSheet("background:#2a2a35; border-radius:6px; border:1px solid #3a3a4a;");
    
    QLabel *imgLabel = new QLabel(previewWidget);
    imgLabel->setGeometry(2, 2, 80, 80);
    imgLabel->setPixmap(result.thumbnail);
    imgLabel->setAlignment(Qt::AlignCenter);

    QPushButton *btnRemove = new QPushButton("×", previewWidget);
    btnRemove->setGeometry(64, 2, 18, 18);
    btnRemove->setStyleSheet("QPushButton { background:rgba(0,0,0,150); color:white; border-radius:9px; font-weight:bold; font-size:12px; padding-bottom:2px; }"
                             "QPushButton:hover { background:rgba(255,50,50,200); }");
    
    connect(btnRemove, &QPushButton::clicked, this, [this, result, previewWidget]() {
        removeAttachment(result.destPath, previewWidget);
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

#include "../../utils/HtmlUtilities.h"

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
    ChatMessageRenderer::renderChatHistory(m_dockUI->chatHistory(), m_ctx, history, currentSessionThinking);
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
