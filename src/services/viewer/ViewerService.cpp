#include "ViewerService.h"
#include "ISceneService.h"
#include "Image2DLoader.h"
#include <QFileInfo>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkRenderWindow.h>

ViewerService::ViewerService(IAppContext* ctx, QObject* parent) 
    : QObject(parent), m_ctx(ctx), currentImageIndex(-1), currentAIMode(AIMode::None), isAutoNext(true) 
{
    autoTimer = new QTimer(this);
    connect(autoTimer, &QTimer::timeout, this, &ViewerService::onAutoTimerTimeout);
}

QString ViewerService::getCurrent2DImagePath() const { 
    return current2DImagePath; 
}

void ViewerService::setCurrent2DImagePath(const QString &path) { 
    current2DImagePath = path; 
}

void ViewerService::setImageList(const QStringList &list, int index) {
    imageFileList = list;
    currentImageIndex = index;
    emit m_ctx->signalBus()->imageListUpdated(imageFileList, currentImageIndex);
    emit m_ctx->signalBus()->imageIndexChanged(currentImageIndex, imageFileList.size());
}

void ViewerService::loadCurrentIndexImage() {
    if (currentImageIndex < 0 || currentImageIndex >= imageFileList.size()) return;
    current2DImagePath = QFileInfo(current2DImagePath).absolutePath() + "/" + imageFileList[currentImageIndex];
    
    if (m_updateScene) {
        m_ctx->scene()->setTextureActor(Image2DLoader::load(current2DImagePath));
        m_ctx->scene()->vtkWidget()->renderWindow()->Render();
    }
    
    emit m_ctx->signalBus()->imageIndexChanged(currentImageIndex, imageFileList.size());
}

void ViewerService::onNextImage() {
    if (currentImageIndex < imageFileList.size() - 1) {
        currentImageIndex++;
        loadCurrentIndexImage();
    }
}

void ViewerService::onPrevImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        loadCurrentIndexImage();
    }
}

void ViewerService::onAutoNext() {
    if (autoTimer->isActive() && isAutoNext) {
        autoTimer->stop();
    } else {
        isAutoNext = true;
        autoTimer->start(500);
    }
    emit m_ctx->signalBus()->autoNavigationChanged(autoTimer->isActive(), isAutoNext);
}

void ViewerService::onAutoPrev() {
    if (autoTimer->isActive() && !isAutoNext) {
        autoTimer->stop();
    } else {
        isAutoNext = false;
        autoTimer->start(500);
    }
    emit m_ctx->signalBus()->autoNavigationChanged(autoTimer->isActive(), isAutoNext);
}

QString ViewerService::getCurrentAIMode() const { 
    return currentAIMode; 
}

void ViewerService::setAIMode(const QString& mode) { 
    currentAIMode = mode; 
}

void ViewerService::onAutoTimerTimeout() {
    if (isAutoNext) {
        if (currentImageIndex < imageFileList.size() - 1) onNextImage();
        else autoTimer->stop();
    } else {
        if (currentImageIndex > 0) onPrevImage();
        else autoTimer->stop();
    }
    if (!autoTimer->isActive()) emit m_ctx->signalBus()->autoNavigationChanged(false, isAutoNext);
}
