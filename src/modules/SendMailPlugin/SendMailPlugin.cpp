#include "SendMailPlugin.h"

#include "IAppContext.h"
#include "MailDockWidget.h"
#include "SendMailRibbonUI.h"
#include "SignalBus.h"

#include <QGroupBox>
#include <QLabel>
#include <QToolButton>
#include <QDockWidget>
#include <QApplication>
#include <QMainWindow>

void SendMailPlugin::initialize(IAppContext *context)
{
    m_ctx = context;
    m_dockUI = new MailDockWidget(m_ctx, this);

    if (QWidget *panel = m_ctx->getTabPanel("tab.mail")) {
        m_ribbonUI = new SendMailRibbonUI(m_ctx, panel, this);
        
        // Initialize button text
        m_ribbonUI->btnToggleMail()->setText(m_ctx->translate("mail.open"));
        
        // Toggle mail visibility
        connect(m_ribbonUI->btnToggleMail(), &QToolButton::clicked, m_dockUI, [this]() {
            if (m_dockUI->dockWidget()->isHidden()) {
                m_dockUI->dockWidget()->show();
            } else {
                m_dockUI->dockWidget()->hide();
            }
        });
        
        // Update button text when visibility changes
        connect(m_dockUI->dockWidget(), &QDockWidget::visibilityChanged, this, [this](bool visible) {
            if (m_ribbonUI) {
                m_ribbonUI->btnToggleMail()->setText(visible ? m_ctx->translate("mail.close") : m_ctx->translate("mail.open"));
            }
        });
        
        connect(m_ribbonUI->btnSettings(), &QToolButton::clicked, m_dockUI, &MailDockWidget::showSettingsDialog);
    }

    connect(m_ctx->signalBus(), &SignalBus::languageChanged, this, [this](const QString &) {
        if (m_dockUI) m_dockUI->retranslate();
        if (!m_ribbonUI) return;
        bool visible = m_dockUI->dockWidget()->isVisible();
        m_ribbonUI->btnToggleMail()->setText(visible ? m_ctx->translate("mail.close") : m_ctx->translate("mail.open"));
        m_ribbonUI->btnSettings()->setText(m_ctx->translate("mail.settings"));
        if (QLabel *label = m_ribbonUI->groupMail()->findChild<QLabel*>("groupTitleLabel")) {
            label->setText(m_ctx->translate("mail.menu"));
        }
    });
    connect(m_ctx->signalBus(), &SignalBus::agentUiActionRequested, this,
            [this](const QString &action, const QVariantMap &parameters) {
        if (!m_dockUI) return;
        bool handled = false;
        if (action == "mail.open") {
            m_dockUI->showInbox();
            handled = true;
        } else if (action == "mail.close" && !m_dockUI->dockWidget()->isHidden()) {
            m_dockUI->dockWidget()->hide();
            handled = true;
        } else if (action == "mail.settings") {
            m_dockUI->showSettingsDialog();
            handled = true;
        }
        const QString requestId = parameters.value("request_id").toString();
        if (!requestId.isEmpty() && action.startsWith("mail."))
            emit m_ctx->signalBus()->agentUiActionCompleted(requestId, handled,
                QVariantMap{{"action", action}, {"error", handled ? "" : "Mail UI is unavailable"}});
    });
}

void SendMailPlugin::cleanup()
{
}
