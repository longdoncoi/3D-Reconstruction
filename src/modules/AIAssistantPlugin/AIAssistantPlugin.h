#ifndef AI_ASSISTANT_PLUGIN_H
#define AI_ASSISTANT_PLUGIN_H

#include <QObject>
#include <QPointer>
#include <QStringList>
#include <QUrl>
#include <QToolButton>
#include <QGroupBox>
#include "IPlugin.h"

#include "IAppContext.h"
class IAIAssistantService;
class AIAssistantRibbonUI;
class ChatBotDockWidget;
class QListWidgetItem;
class QWidget;

class AIAssistantPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "AI Assistant Plugin"; }
    void initialize(IAppContext* context) override;
    void cleanup() override;
    int  loadOrder() const override { return 40; }

private slots:
    void onToggleChatbot();
    void onNewChat();
    void onToggleSessionHistory();
    void onSendChatMessage();
    void onModelSelected(int index);
    void onChatLinkClicked(const QUrl &url);
    void updateSessionListUI();
    void onSessionClicked(QListWidgetItem *item);
    void onSessionMenuRequested(const QPoint &pos);
    void onAssistantStatusChanged(const QString &status);
    void onAssistantError(const QString &error);
    void onAttachImage();
    void onAttachFile();
    void removeAttachment(const QString &filePath, QWidget *previewWidget);
    void updateChatUI();
    void onProgressStopped();

private:
    bool m_isStartingServer = false;
    bool m_pendingNewChat = false;  // Track if we should create new session on next send
    CustomProgressDialog* m_progressDialog = nullptr;
    void setupChatbotUI();
    
    QString buildMessageHtml(const QString &role, const QString &content, const QJsonArray &attachments);
    void addAttachmentPreview(const QString &filePath, bool isImage);

    QStringList pendingAttachments;

    IAppContext* m_ctx = nullptr;
    IAIAssistantService* m_aiAssistant = nullptr;

    AIAssistantRibbonUI* m_ribbonUI = nullptr;
    ChatBotDockWidget* m_dockUI = nullptr;
};

#endif // AI_ASSISTANT_PLUGIN_H
