#include "AIProcessorRibbonUI.h"
#include "IAppContext.h"
#include "IconFactory.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QToolButton>
#include <QLabel>

AIProcessorRibbonUI::AIProcessorRibbonUI(IAppContext* ctx, QWidget* panel, QObject* parent) 
    : QObject(parent), m_ctx(ctx) 
{
    auto *layout = qobject_cast<QHBoxLayout*>(panel->layout());
    if (!layout) return;

    m_groupAI = new QGroupBox(panel);
    m_groupAI->setObjectName("aiProcessingGroup");
    m_groupAI->setTitle("");

    QVBoxLayout *vbox1 = new QVBoxLayout(m_groupAI);
    vbox1->setContentsMargins(4, 4, 4, 4);
    vbox1->setSpacing(2);

    QHBoxLayout *gLayout = new QHBoxLayout();
    gLayout->setContentsMargins(0, 0, 0, 0);
    gLayout->setSpacing(5);

    m_btnDet = new QToolButton(m_groupAI);
    m_btnDet->setText(m_ctx->translate("ai.run_detection"));
    m_btnDet->setIcon(IconFactory::createModern("🔍", QColor("#f59e0b"), QColor("#d97706")));
    m_btnDet->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnDet->setMinimumWidth(80);

    m_btnSeg = new QToolButton(m_groupAI);
    m_btnSeg->setText(m_ctx->translate("ai.run_segmentation"));
    m_btnSeg->setIcon(IconFactory::createModern("🎯", QColor("#10b981"), QColor("#059669")));
    m_btnSeg->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnSeg->setMinimumWidth(80);

    m_btnHide = new QToolButton(m_groupAI);
    m_btnHide->setText(m_ctx->translate("ai.hide_results"));
    m_btnHide->setIcon(IconFactory::createModern("👁️", QColor("#6b7280"), QColor("#4b5563")));
    m_btnHide->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnHide->setMinimumWidth(80);

    m_btnTrack = new QToolButton(m_groupAI);
    m_btnTrack->setText(m_ctx->translate("ai.video_tracking"));
    m_btnTrack->setIcon(IconFactory::createModern("🎥", QColor("#ef4444"), QColor("#b91c1c")));
    m_btnTrack->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnTrack->setMinimumWidth(80);

    m_btnPauseTrack = new QToolButton(m_groupAI);
    m_btnPauseTrack->setText("Pause");
    m_btnPauseTrack->setIcon(IconFactory::createModern("⏸️", QColor("#f59e0b"), QColor("#d97706")));
    m_btnPauseTrack->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnPauseTrack->setMinimumWidth(80);
    m_btnPauseTrack->setVisible(false);

    m_btnStopTrack = new QToolButton(m_groupAI);
    m_btnStopTrack->setText("Stop");
    m_btnStopTrack->setIcon(IconFactory::createModern("⏹️", QColor("#ef4444"), QColor("#b91c1c")));
    m_btnStopTrack->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnStopTrack->setMinimumWidth(80);
    m_btnStopTrack->setVisible(false);

    gLayout->addWidget(m_btnDet);
    gLayout->addWidget(m_btnSeg);
    gLayout->addWidget(m_btnTrack);
    gLayout->addWidget(m_btnPauseTrack);
    gLayout->addWidget(m_btnStopTrack);
    gLayout->addWidget(m_btnHide);
    vbox1->addLayout(gLayout);
    vbox1->addStretch();

    QLabel *titleLabel1 = new QLabel(m_ctx->translate("ai.menu"), m_groupAI);
    titleLabel1->setObjectName("groupTitleLabel");
    titleLabel1->setAlignment(Qt::AlignCenter);
    titleLabel1->setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;");
    vbox1->addWidget(titleLabel1);

    m_groupTrain = new QGroupBox(panel);
    m_groupTrain->setObjectName("aiTrainingGroup");
    m_groupTrain->setTitle("");

    QVBoxLayout *vbox2 = new QVBoxLayout(m_groupTrain);
    vbox2->setContentsMargins(4, 4, 4, 4);
    vbox2->setSpacing(2);

    QHBoxLayout *tLayout = new QHBoxLayout();
    tLayout->setContentsMargins(0, 0, 0, 0);
    tLayout->setSpacing(5);

    m_btnTrain = new QToolButton(m_groupTrain);
    m_btnTrain->setText(m_ctx->translate("ai.training"));
    m_btnTrain->setIcon(IconFactory::createModern("⚙️", QColor("#8b5cf6"), QColor("#7c3aed")));
    m_btnTrain->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnTrain->setMinimumWidth(80);

    m_btnChart = new QToolButton(m_groupTrain);
    m_btnChart->setText(m_ctx->translate("ai.charts"));
    m_btnChart->setIcon(IconFactory::createModern("📊", QColor("#06b6d4"), QColor("#0891b2")));
    m_btnChart->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_btnChart->setMinimumWidth(80);

    tLayout->addWidget(m_btnTrain);
    tLayout->addWidget(m_btnChart);
    vbox2->addLayout(tLayout);
    vbox2->addStretch();

    QLabel *titleLabel2 = new QLabel(m_ctx->translate("ai.training"), m_groupTrain);
    titleLabel2->setObjectName("groupTitleLabel");
    titleLabel2->setAlignment(Qt::AlignCenter);
    titleLabel2->setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;");
    vbox2->addWidget(titleLabel2);

    layout->insertWidget(layout->count() - 1, m_groupAI);
    layout->insertWidget(layout->count() - 1, m_groupTrain);
}
