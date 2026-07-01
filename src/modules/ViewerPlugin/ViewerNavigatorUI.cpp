#include "ViewerNavigatorUI.h"
#include "IAppContext.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>

ViewerNavigatorUI::ViewerNavigatorUI(IAppContext* ctx, QObject* parent)
    : QObject(parent), m_ctx(ctx)
{
    m_widget = new QWidget(m_ctx->mainWindow()); 
    m_widget->setObjectName("viewerNavigatorWidget");
    QHBoxLayout *nl = new QHBoxLayout(m_widget);
    m_widget->setFixedHeight(80);
    
    m_btnPrev = new QPushButton(m_ctx->translate("viewer.prev"), m_ctx->mainWindow());
    m_btnPrev->setMinimumWidth(80);
    m_btnAutoPrev = new QPushButton(m_ctx->translate("viewer.auto_prev"), m_ctx->mainWindow());
    m_btnAutoPrev->setMinimumWidth(100);
    m_btnAutoNext = new QPushButton(m_ctx->translate("viewer.auto_next"), m_ctx->mainWindow());
    m_btnAutoNext->setMinimumWidth(100);
    m_btnNext = new QPushButton(m_ctx->translate("viewer.next"), m_ctx->mainWindow());
    m_btnNext->setMinimumWidth(80);
    
    m_btnPrev->setEnabled(false);
    m_btnAutoPrev->setEnabled(false);
    m_btnAutoNext->setEnabled(false);
    m_btnNext->setEnabled(false);
    
    QString style = "QPushButton { padding: 4px; font-size: 13px; }";
    m_btnPrev->setStyleSheet(style);
    m_btnAutoPrev->setStyleSheet(style);
    m_btnAutoNext->setStyleSheet(style);
    m_btnNext->setStyleSheet(style);
    
    m_btnAutoPrev->setCheckable(true); 
    m_btnAutoNext->setCheckable(true);
    
    nl->addStretch(); 
    nl->addWidget(m_btnPrev); 
    nl->addWidget(m_btnAutoPrev); 
    nl->addWidget(m_btnAutoNext); 
    nl->addWidget(m_btnNext); 
    nl->addStretch();
    
    if (m_ctx->mainWindow()->centralWidget() && m_ctx->mainWindow()->centralWidget()->layout()) {
        qobject_cast<QVBoxLayout*>(m_ctx->mainWindow()->centralWidget()->layout())->addWidget(m_widget);
    }
}
