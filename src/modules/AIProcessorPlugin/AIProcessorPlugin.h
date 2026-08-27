#ifndef AI_PROCESSOR_PLUGIN_H
#define AI_PROCESSOR_PLUGIN_H

#include <QObject>
#include "IPlugin.h"
#include <QAction>
#include <QProcess>
#include <QToolButton>
#include <QGroupBox>
#include <QFutureWatcher>
#include <atomic>

#include "IAppContext.h"
#include "IAIService.h"
#include <QTextBrowser>
#include <QProgressBar>
#include <QPushButton>

class AITrainDockWidget;
class AIProcessorRibbonUI;
class VideoTrackerThread;
class QTimer;

class AIProcessorPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    QString pluginName() const override { return "AI Processor Plugin"; }
    void initialize(IAppContext* context) override;
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
    CustomProgressDialog *m_progressDialog = nullptr;
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

    // Train AI UI
    AITrainDockWidget* m_trainDock = nullptr;

    // Tab buttons
    AIProcessorRibbonUI* m_ribbonUI = nullptr;

    // Async model loading
    QFutureWatcher<void>* m_modelLoadWatcher = nullptr;
    bool m_modelsLoading = false;
    std::atomic<bool> m_cancelModelLoad{false};
    QMetaObject::Connection m_modelLoadStopConn;

    VideoTrackerThread* m_trackerThread = nullptr;
    bool m_trackingPaused = false;
    QString m_agentVideoPath;
};

#endif // AI_PROCESSOR_PLUGIN_H
