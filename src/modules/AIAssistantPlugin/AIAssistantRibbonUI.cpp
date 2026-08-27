#include "AIAssistantRibbonUI.h"
#include "IconFactory.h"

AIAssistantRibbonUI::AIAssistantRibbonUI(IAppContext* ctx, QWidget* parentPanel, QObject* parent)
    : QObject(parent), m_ctx(ctx) 
{
    auto *layout = qobject_cast<QHBoxLayout*>(parentPanel->layout());
    if (layout) {
        m_groupAI = new QGroupBox(parentPanel);
        m_groupAI->setObjectName("aiAssistantGroup");
        m_groupAI->setTitle("");

        QVBoxLayout *vbox = new QVBoxLayout(m_groupAI);
        vbox->setContentsMargins(4, 4, 4, 4);
        vbox->setSpacing(2);

        QHBoxLayout *gLayout = new QHBoxLayout();
        gLayout->setContentsMargins(0, 0, 0, 0);
        gLayout->setSpacing(5);

        m_btnToggleAssistant = new QToolButton(m_groupAI);
        m_btnToggleAssistant->setText(m_ctx->translate("ai.open_assistant"));
        m_btnToggleAssistant->setIcon(IconFactory::createModern("💬", QColor("#10b981"), QColor("#059669")));
        m_btnToggleAssistant->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_btnToggleAssistant->setMinimumWidth(100);

        gLayout->addWidget(m_btnToggleAssistant);

        m_btnRestartModel = new QToolButton(m_groupAI);
        m_btnRestartModel->setText(m_ctx->translate("ai.restart_model"));
        m_btnRestartModel->setIcon(IconFactory::createModern("🔄", QColor("#3b82f6"), QColor("#2563eb")));
        m_btnRestartModel->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_btnRestartModel->setMinimumWidth(80);
        gLayout->addWidget(m_btnRestartModel);

        m_btnRestartRAG = new QToolButton(m_groupAI);
        m_btnRestartRAG->setText(m_ctx->translate("ai.restart_rag"));
        m_btnRestartRAG->setIcon(IconFactory::createModern("📚", QColor("#eab308"), QColor("#ca8a04")));
        m_btnRestartRAG->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_btnRestartRAG->setMinimumWidth(80);
        gLayout->addWidget(m_btnRestartRAG);

        m_btnRestartAgent = new QToolButton(m_groupAI);
        m_btnRestartAgent->setText(m_ctx->translate("ai.restart_agent"));
        m_btnRestartAgent->setIcon(IconFactory::createModern("🤖", QColor("#8b5cf6"), QColor("#7c3aed")));
        m_btnRestartAgent->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_btnRestartAgent->setMinimumWidth(80);
        gLayout->addWidget(m_btnRestartAgent);

        m_btnRestartServer = new QToolButton(m_groupAI);
        m_btnRestartServer->setText(m_ctx->translate("ai.restart_server"));
        m_btnRestartServer->setIcon(IconFactory::createModern("🖥️", QColor("#ef4444"), QColor("#b91c1c")));
        m_btnRestartServer->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_btnRestartServer->setMinimumWidth(80);
        gLayout->addWidget(m_btnRestartServer);

        vbox->addLayout(gLayout);
        vbox->addStretch();

        QLabel *titleLabel = new QLabel(m_ctx->translate("menu.ai_assistant"), m_groupAI);
        titleLabel->setObjectName("groupTitleLabel");
        titleLabel->setAlignment(Qt::AlignCenter);
        titleLabel->setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;");
        vbox->addWidget(titleLabel);

        layout->insertWidget(layout->count() - 1, m_groupAI);
    }
}

