#ifndef VIEWER_PLUGIN_H
#define VIEWER_PLUGIN_H

#include <QObject>
#include <QStringList>
#include "IPlugin.h"
#include "AppConstants.h"
#include <QAction>
#include <QToolButton>
#include <QGroupBox>

#include "IAppContext.h"
class QPushButton;

class IProgressDialog;
class ViewerRibbonUI;
class ViewerNavigatorUI;
class ViewerListUI;
class ViewerViewModel;

class ViewerPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "Viewer Plugin"; }
    QStringList dependencies() const override {
        return {QString::fromLatin1(AppConstants::PluginNames::UserAuthentication)};
    }
    bool initialize(IAppContext* context) override;
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
    void load2DImageFromPath(const QString &filePath);
    void load3DModelFromPath(const QString &filePath);
    void loadDicomFromDirectory(const QString &directory);

    IAppContext* m_ctx = nullptr;
    
    QAction *m_load2DAct = nullptr;
    QAction *m_load3DAct = nullptr;
    QAction *m_loadDicomAct = nullptr;

    ViewerNavigatorUI *m_navUI = nullptr;
    ViewerRibbonUI *m_ribbonUI = nullptr;
    ViewerListUI *m_listUI = nullptr;
    IProgressDialog *m_progressDialog = nullptr;
    ViewerViewModel *m_viewModel = nullptr;
};

#endif // VIEWER_PLUGIN_H
