#ifndef RECONSTRUCTION_PLUGIN_H
#define RECONSTRUCTION_PLUGIN_H

#include <QObject>
#include <QStringList>
#include "IPlugin.h"
#include "AppConstants.h"
#include "IAppContext.h"
#include "IReconstructionService.h"
#include <QAction>
#include <QToolButton>
#include <QListWidget>

class IProgressDialog;
class ReconstructionRibbonUI;

class ReconstructionPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "Reconstruction Plugin"; }
    QStringList dependencies() const override {
        return {QString::fromLatin1(AppConstants::PluginNames::UserAuthentication)};
    }
    bool initialize(IAppContext* context) override;
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
    IProgressDialog* m_progressDialog = nullptr;

    QMenu* m_reconstructMenu = nullptr;
    bool m_cancelRequested = false;  ///< true while a stop-requested run is finishing
    
    QAction *m_toggleCloudAct = nullptr;
    ReconstructionRibbonUI* m_ribbonUI = nullptr;
    QString m_agentImageFolder;
    bool m_skipImageLoadForAgentRun = false;
};

#endif // RECONSTRUCTION_PLUGIN_H
