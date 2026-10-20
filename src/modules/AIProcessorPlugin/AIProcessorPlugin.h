#ifndef AI_PROCESSOR_PLUGIN_H
#define AI_PROCESSOR_PLUGIN_H

#include <QObject>
#include <QStringList>
#include "IPlugin.h"
#include "AppConstants.h"
#include <QAction>
#include <QProcess>
#include <QToolButton>
#include <QGroupBox>

#include "IAppContext.h"
#include "IAIService.h"

class IProgressDialog;
#include <QTextBrowser>
#include <QProgressBar>
#include <QPushButton>

class AITrainDockWidget;
class AIProcessorRibbonUI;
class VideoTrackerThread;
class IAIModelLoadService;
class QTimer;
class QMenu;

class AIProcessorPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "AI Processor Plugin"; }
    QStringList dependencies() const override {
        return {QString::fromLatin1(AppConstants::PluginNames::UserAuthentication)};
    }
    bool initialize(IAppContext* context) override;
    void cleanup() override;
    int  loadOrder() const override { return 30; }

private slots:
    void onTrainModel();
    void onObjectDetection();
    void onSegmentation();
    void onHideAIResults();
    void onObjectTracking();
    void onPauseResumeTracking();
    void onStopTracking();
    void onTrackingFrameReceived(const cv::Mat& frame);
    void onTrackingFinished();
    void onViewCharts();
    void updateActions();
    void onImageChanged(int index, int total);

private:
    void loadModelsInBackground();
    void setupMenus();
    void setupRibbonUI();
    void setupConnections();

    IAppContext*   m_ctx         = nullptr;
    IAIService*    m_aiSvc       = nullptr;  ///< Looked up from ServiceRegistry
    IAIModelLoadService* m_modelLoader = nullptr; ///< AppCore-owned async loader
    IProgressDialog *m_progressDialog = nullptr;
    QProcess *m_tensorboardProcess = nullptr;
    QTimer *m_tensorboardWaitTimer = nullptr;

    // Viewport and renderers;
    
    QAction *m_runDetAct = nullptr;
    QAction *m_hideDetAct = nullptr;
    QAction *m_runSegAct = nullptr;
    QAction *m_hideSegAct = nullptr;
    QAction *m_runTrackingAct = nullptr;
    QAction *m_pauseTrackingAct = nullptr;
    QAction *m_stopTrackingAct = nullptr;

    // Sub-menus kept so titles can be re-applied on language change
    QMenu *m_detMenu = nullptr;
    QMenu *m_segMenu = nullptr;
    QMenu *m_trackMenu = nullptr;

    // Train AI UI
    AITrainDockWidget* m_trainDock = nullptr;

    // Tab buttons
    AIProcessorRibbonUI* m_ribbonUI = nullptr;

    // Async model loading is owned by IAIModelLoadService (AppCore).
    bool m_modelsLoading = false;

    VideoTrackerThread* m_trackerThread = nullptr;
    bool m_trackingPaused = false;
    QString m_agentVideoPath;
};

#endif // AI_PROCESSOR_PLUGIN_H
