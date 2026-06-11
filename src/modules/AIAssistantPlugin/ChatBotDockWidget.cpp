#include "ChatBotDockWidget.h"
#include "AppConstants.h"
#include "UserManager.h"
#include "IconFactory.h"
#include <QVBoxLayout>
#include <QMenu>

ChatBotDockWidget::ChatBotDockWidget(IAppContext* ctx, QObject* parent)
    : QObject(parent), m_ctx(ctx)
{
    m_dockWidget = new QDockWidget(m_ctx->translate("ai.dock_title"), m_ctx->mainWindow());
    m_dockWidget->setMinimumWidth(AppConstants::AIAssistant::DEFAULT_DOCK_WIDTH);
    QWidget* emptyTitle = new QWidget();
    emptyTitle->setFixedHeight(0);
    m_dockWidget->setTitleBarWidget(emptyTitle); // Hide native title bar to use custom one below

    QWidget     *dw = new QWidget(m_dockWidget);
    dw->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *dl = new QVBoxLayout(dw);
    dl->setContentsMargins(0, 0, 0, 0);
    dl->setSpacing(0);

    dl->addWidget(createTopBar(dw));

    m_chatSplitter = new QSplitter(Qt::Horizontal, dw);
    m_chatSplitter->setHandleWidth(8);
    m_chatSplitter->setChildrenCollapsible(true);
    m_chatSplitter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    m_chatSplitter->addWidget(createSessionPanel(m_chatSplitter));
    m_chatSplitter->addWidget(createChatArea(m_chatSplitter));
    m_chatSplitter->setCollapsible(0, true);
    m_chatSplitter->setCollapsible(1, false);

    m_chatSplitter->setSizes({200, 10000});
    m_chatSplitter->setStretchFactor(0, 0);
    m_chatSplitter->setStretchFactor(1, 1);

    dl->addWidget(m_chatSplitter, 1);

    m_dockWidget->setWidget(dw);
    m_ctx->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, m_dockWidget);
    m_dockWidget->hide();
}

QWidget* ChatBotDockWidget::createTopBar(QWidget* parent) {
    QWidget     *topBar = new QWidget(parent);
    topBar->setObjectName("modernTitleBar");
    QHBoxLayout *tl     = new QHBoxLayout(topBar);
    tl->setContentsMargins(8, 6, 8, 6);
    tl->setSpacing(6);
    topBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    m_btnToggleHistory = new QToolButton(topBar);
    m_btnToggleHistory->setText("☰");
    m_btnToggleHistory->setObjectName("modernControlBtn");
    tl->addWidget(m_btnToggleHistory);

    m_modelSelector = new QComboBox(topBar);
    m_modelSelector->addItems(AppConstants::AIAssistant::modelNames());
    
    // Load per-user model index
    int savedIdx = 0;
    if (auto *um = UserManager::instance()) {
        QString username = um->currentUsername();
        savedIdx = um->getUserPref(username, "ai_model_index", "0").toInt();
    }
    m_modelSelector->setCurrentIndex(savedIdx);
    tl->addWidget(m_modelSelector, 1);

    m_btnNewChat = new QPushButton(m_ctx->translate("ai.new_chat"), topBar);
    m_btnNewChat->setObjectName("primary");
    tl->addWidget(m_btnNewChat);

    return topBar;
}

QWidget* ChatBotDockWidget::createSessionPanel(QWidget* parent) {
    m_sessionPanel = new QWidget(parent);
    m_sessionPanel->setObjectName("sessionPanel");
    m_sessionPanel->setMinimumWidth(AppConstants::AIAssistant::SESSION_PANEL_MIN_WIDTH);
    m_sessionPanel->setMaximumWidth(AppConstants::AIAssistant::SESSION_PANEL_MAX_WIDTH);
    m_sessionPanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    QVBoxLayout *sl = new QVBoxLayout(m_sessionPanel);
    sl->setContentsMargins(0, 8, 0, 8);
    sl->setSpacing(2);

    m_sessLabel = new QLabel(m_ctx->translate("ai.recent"), m_sessionPanel);
    m_sessLabel->setStyleSheet("color:#888; font-size:12px; padding:4px 8px 8px 8px;");
    sl->addWidget(m_sessLabel);

    m_sessionListWidget = new QListWidget(m_sessionPanel);
    m_sessionListWidget->setObjectName("sessionList");
    m_sessionListWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_sessionListWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    m_sessionListWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);  // Enable multi-select
    sl->addWidget(m_sessionListWidget, 1);
    
    return m_sessionPanel;
}

QWidget* ChatBotDockWidget::createChatArea(QWidget* parent) {
    QWidget     *chatArea = new QWidget(parent);
    chatArea->setObjectName("mainContainer");
    QVBoxLayout *cl       = new QVBoxLayout(chatArea);
    cl->setContentsMargins(0, 0, 0, 0);
    cl->setSpacing(0);
    chatArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    m_chatHistory = new QTextBrowser(chatArea);
    m_chatHistory->setObjectName("chatHistory");
    m_chatHistory->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_chatHistory->setOpenLinks(false);
    cl->addWidget(m_chatHistory, 1);

    m_attachmentPreviewArea = new QWidget(chatArea);
    m_attachmentPreviewArea->setObjectName("attachmentArea");
    m_attachmentPreviewArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_attachmentLayout = new QHBoxLayout(m_attachmentPreviewArea);
    m_attachmentLayout->setContentsMargins(8, 8, 8, 8);
    m_attachmentLayout->setSpacing(8);
    m_attachmentLayout->setAlignment(Qt::AlignLeft);
    m_attachmentPreviewArea->hide();
    cl->addWidget(m_attachmentPreviewArea, 0);

    cl->addWidget(createInputRow(chatArea));
    
    return chatArea;
}

QWidget* ChatBotDockWidget::createInputRow(QWidget* parent) {
    QWidget     *inputRow = new QWidget(parent);
    inputRow->setObjectName("modernTitleBar");
    QHBoxLayout *il       = new QHBoxLayout(inputRow);
    inputRow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    il->setContentsMargins(8, 6, 8, 6);
    il->setSpacing(6);

    m_btnAttach = new QToolButton(inputRow);
    m_btnAttach->setText("+");
    m_btnAttach->setToolTip(m_ctx->translate("ai.attach"));
    m_btnAttach->setFixedSize(40, 40);
    m_btnAttach->setObjectName("modernControlBtn");
    m_btnAttach->setPopupMode(QToolButton::InstantPopup);
    m_btnAttach->setStyleSheet("QToolButton::menu-indicator { image: none; }");

    QMenu *attachMenu = new QMenu(m_btnAttach);
    m_actAttachImage = attachMenu->addAction(
        IconFactory::createModern("🖼️", QColor("#3b82f6"), QColor("#1d4ed8")),
        m_ctx->translate("ai.attach_image"));
    m_actAttachFile  = attachMenu->addAction(
        IconFactory::createModern("📄", QColor("#a855f7"), QColor("#7e22ce")),
        m_ctx->translate("ai.attach_file"));
    m_btnAttach->setMenu(attachMenu);

    m_chatInput = new QLineEdit(inputRow);
    m_chatInput->setPlaceholderText(m_ctx->translate("ai.input_hint"));
    m_btnSendChat = new QPushButton("➤", inputRow);
    m_btnSendChat->setObjectName("primary");
    m_btnSendChat->setFixedSize(40, 40);

    il->addWidget(m_btnAttach);
    il->addWidget(m_chatInput);
    il->addWidget(m_btnSendChat);
    
    return inputRow;
}
