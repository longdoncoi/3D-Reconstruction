#ifndef VIEWERSERVICE_H
#define VIEWERSERVICE_H

#include "IAppContext.h"
#include "IViewerService.h"
#include "SignalBus.h"
#include <QTimer>

class ViewerService : public QObject, public IViewerService {
    Q_OBJECT
public:
    ViewerService(IAppContext* ctx, QObject* parent = nullptr);

    QString getCurrent2DImagePath() const override;
    void setCurrent2DImagePath(const QString &path) override;
    void setImageList(const QStringList &list, int index) override;
    void loadCurrentIndexImage() override;
    void onNextImage() override;
    void onPrevImage() override;
    void onAutoNext() override;
    void onAutoPrev() override;
    QString getCurrentAIMode() const override;
    void setAIMode(const QString& mode) override;

    bool isUpdateSceneEnabled() const override { return m_updateScene; }
    void setUpdateSceneEnabled(bool enabled) override { m_updateScene = enabled; }

private slots:
    void onAutoTimerTimeout();

private:
    IAppContext* m_ctx;
    QString current2DImagePath;
    QStringList imageFileList;
    int currentImageIndex;
    QString currentAIMode;
    QTimer *autoTimer;
    bool isAutoNext;
    bool m_updateScene = true;
};

#endif // VIEWERSERVICE_H
