#ifndef CHAT_BOT_DOCK_WIDGET_H
#define CHAT_BOT_DOCK_WIDGET_H

#include <QObject>
#include <QDockWidget>
#include <QSplitter>
#include <QListWidget>
#include <QToolButton>
#include <QTextBrowser>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QComboBox>
#include <QLabel>
#include <QAction>

#include "IAppContext.h"

class ChatBotDockWidget : public QObject {
    Q_OBJECT
public:
    explicit ChatBotDockWidget(IAppContext* ctx, QObject* parent = nullptr);

    QDockWidget* dockWidget() const { return m_dockWidget; }
    QSplitter* chatSplitter() const { return m_chatSplitter; }
    QWidget* sessionPanel() const { return m_sessionPanel; }
    QListWidget* sessionListWidget() const { return m_sessionListWidget; }
    QToolButton* btnToggleHistory() const { return m_btnToggleHistory; }
    QTextBrowser* chatHistory() const { return m_chatHistory; }
    QToolButton* btnAttach() const { return m_btnAttach; }
    QWidget* attachmentPreviewArea() const { return m_attachmentPreviewArea; }
    QHBoxLayout* attachmentLayout() const { return m_attachmentLayout; }
    QLineEdit* chatInput() const { return m_chatInput; }
    QPushButton* btnSendChat() const { return m_btnSendChat; }
    QPushButton* btnNewChat() const { return m_btnNewChat; }
    QComboBox* modelSelector() const { return m_modelSelector; }
    QLabel* sessLabel() const { return m_sessLabel; }
    QAction* actAttachImage() const { return m_actAttachImage; }
    QAction* actAttachFile() const { return m_actAttachFile; }
    QPushButton* btnToggleAgentMode() const { return m_btnToggleAgentMode; }
    void setAgentModeActive(bool active);

private:
    QWidget* createTopBar(QWidget* parent);
    QWidget* createSessionPanel(QWidget* parent);
    QWidget* createChatArea(QWidget* parent);
    QWidget* createInputRow(QWidget* parent);


    IAppContext* m_ctx;

    QDockWidget*   m_dockWidget = nullptr;
    QSplitter*     m_chatSplitter = nullptr;
    QWidget*       m_sessionPanel = nullptr;
    QListWidget*   m_sessionListWidget = nullptr;
    QToolButton*   m_btnToggleHistory = nullptr;
    QTextBrowser*  m_chatHistory = nullptr;
    QToolButton*   m_btnAttach = nullptr;
    QWidget*       m_attachmentPreviewArea = nullptr;
    QHBoxLayout*   m_attachmentLayout = nullptr;
    QLineEdit*     m_chatInput = nullptr;
    QPushButton*   m_btnSendChat = nullptr;
    QPushButton*   m_btnNewChat = nullptr;
    QComboBox*     m_modelSelector = nullptr;
    QLabel*        m_sessLabel = nullptr;
    QAction*       m_actAttachImage = nullptr;
    QAction*       m_actAttachFile = nullptr;
    QPushButton*   m_btnToggleAgentMode = nullptr;
};

#endif // CHAT_BOT_DOCK_WIDGET_H
