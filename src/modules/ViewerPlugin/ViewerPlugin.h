#ifndef VIEWER_PLUGIN_H
#define VIEWER_PLUGIN_H

#include <QObject>
#include "IPlugin.h"
#include <QAction>
#include <QToolButton>
#include <QGroupBox>

#include "IAppContext.h"
class QPushButton;

class CustomProgressDialog;
class ViewerRibbonUI;
class ViewerNavigatorUI;
class ViewerListUI;

class ViewerPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "Viewer Plugin"; }
    void initialize(IAppContext* context) override;
    void cleanup() override;
    int  loadOrder() const override { return 10; }

private slots:
    void onLoad2DImages();
    void onLoad3DImages();
    void onLoadDicom();
    void onPrevImage();
    void onNextImage();
    void onAutoPrev();
    void onAutoNext();
    void updateActions();
    void onImageIndexChanged(int index, int total);
    void onAutoNavigationChanged(bool active, bool isNext);

private:
    IAppContext* m_ctx = nullptr;
    
    QAction *m_load2DAct = nullptr;
    QAction *m_load3DAct = nullptr;
    QAction *m_loadDicomAct = nullptr;

    ViewerNavigatorUI *m_navUI = nullptr;
    ViewerRibbonUI *m_ribbonUI = nullptr;
    ViewerListUI *m_listUI = nullptr;
    CustomProgressDialog *m_progressDialog = nullptr;
};

#endif // VIEWER_PLUGIN_H
