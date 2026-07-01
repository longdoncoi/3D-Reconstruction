#ifndef RECONSTRUCTION_PLUGIN_H
#define RECONSTRUCTION_PLUGIN_H

#include <QObject>
#include "IPlugin.h"
#include "IAppContext.h"
#include "IReconstructionService.h"
#include <QAction>
#include <QToolButton>
#include <QListWidget>

class ReconstructThread;
class CustomProgressDialog;
class ReconstructionRibbonUI;

class ReconstructionPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "Reconstruction Plugin"; }
    void initialize(IAppContext* context) override;
    void cleanup() override;
    int  loadOrder() const override { return 20; }

private slots:
    void onLoadMultipleImages();
    void onRunReconstruction();
    void onTogglePointCloud(bool forceShow = false);
    void onShowPointCloud();
    void onHidePointCloud();
    void onProgressStopped();
    void updateActions();

private:
    IAppContext*             m_ctx           = nullptr;
    IReconstructionService*  m_reconSvc      = nullptr;  ///< Looked up from ServiceRegistry
    bool m_isPointCloudVisible = false;
    CustomProgressDialog* m_progressDialog = nullptr;

    QMenu* m_reconstructMenu = nullptr;
    ReconstructThread* m_currentReconstructThread = nullptr;
    
    QAction *m_toggleCloudAct = nullptr;
    ReconstructionRibbonUI* m_ribbonUI = nullptr;
};

#endif // RECONSTRUCTION_PLUGIN_H
