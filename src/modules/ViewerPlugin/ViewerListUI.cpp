#include "ViewerListUI.h"
#include "IAppContext.h"
#include "SignalBus.h"
#include "IViewerService.h"
#include <QMainWindow>
#include <QVBoxLayout>
#include <QListWidget>
#include <QFileInfo>
#include <QDir>
#include <QApplication>

ViewerListUI::ViewerListUI(IAppContext* ctx, QObject* parent)
    : QObject(parent), m_ctx(ctx) {
}

void ViewerListUI::setupUI() {
    if (QWidget* central = m_ctx->mainWindow()->centralWidget()) {
        if (QVBoxLayout* cLayout = qobject_cast<QVBoxLayout*>(central->layout())) {
            m_imageList = new QListWidget(central);
            m_imageList->setObjectName("viewerImageList");
            m_imageList->setFixedHeight(120);
            m_imageList->setViewMode(QListView::IconMode);
            m_imageList->setFlow(QListView::LeftToRight);
            m_imageList->setWrapping(false);
            m_imageList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
            m_imageList->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
            m_imageList->setIconSize(QSize(80, 80));
            m_imageList->setResizeMode(QListView::Adjust);
            m_imageList->setSpacing(5);
            m_imageList->hide();
            cLayout->addWidget(m_imageList);
            
            connect(m_imageList, &QListWidget::currentRowChanged, this, &ViewerListUI::onListRowChanged);
            connect(m_ctx->signalBus(), &SignalBus::imageListUpdated, this, &ViewerListUI::onImageListUpdated);
            connect(m_ctx->signalBus(), &SignalBus::imageIndexChanged, this, &ViewerListUI::onImageIndexChanged);
        }
    }
}

void ViewerListUI::onListRowChanged(int row) {
    if (row >= 0 && m_imageList) {
        QStringList filenames;
        for (int i = 0; i < m_imageList->count(); ++i) {
            filenames.append(QFileInfo(m_imageList->item(i)->toolTip()).fileName());
        }
        
        m_ctx->viewer()->setImageList(filenames, row);
        m_ctx->viewer()->setCurrent2DImagePath(m_imageList->item(row)->toolTip());
        m_ctx->viewer()->loadCurrentIndexImage();
    }
}

void ViewerListUI::onImageListUpdated(const QStringList& images, int currentIndex) {
    if (!m_imageList) return;
    m_imageList->blockSignals(true);
    m_imageList->clear();

    QString currentImg = m_ctx->viewer()->getCurrent2DImagePath();
    if (currentImg.isEmpty() || images.isEmpty()) {
        m_imageList->hide();
        m_imageList->blockSignals(false);
        return;
    }

    QDir dir = QFileInfo(currentImg).dir();
    for (const QString& fName : images) {
        QString fullPath = dir.absoluteFilePath(fName);
        QListWidgetItem* item = new QListWidgetItem(QIcon(fullPath), fName);
        item->setToolTip(fullPath);
        m_imageList->addItem(item);
    }
    
    if (currentIndex >= 0 && currentIndex < m_imageList->count()) {
        m_imageList->setCurrentRow(currentIndex);
    }
    
    m_imageList->show();
    m_imageList->blockSignals(false);
}

void ViewerListUI::onImageIndexChanged(int index, int total) {
    Q_UNUSED(total);
    if (!m_imageList) return;
    if (index >= 0 && index < m_imageList->count()) {
        m_imageList->blockSignals(true);
        m_imageList->setCurrentRow(index);
        m_imageList->scrollToItem(m_imageList->item(index), QAbstractItemView::PositionAtCenter);
        m_imageList->blockSignals(false);
    }
}
