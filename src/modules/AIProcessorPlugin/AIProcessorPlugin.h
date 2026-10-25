#ifndef AI_PROCESSOR_PLUGIN_H
#define AI_PROCESSOR_PLUGIN_H

#include <QObject>
#include <QStringList>
#include <functional>
#include "IPlugin.h"
#include "AppConstants.h"
#include <QAction>
#include <QProcess>
#include <QToolButton>
#include <QGroupBox>

#include "IAppContext.h"
#include "IAIService.h"

#include "BlockingCall.h"

class IProgressDialog;
#include <QTextBrowser>
#include <QProgressBar>
#include <QPushButton>

class AITrainDockWidget;
class AIProcessorRibbonUI;
class VideoTrackerThread;
class IAIModelLoadService;
class IAIPredictionLogService;
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
    void onTrackingError(const QString& message);
    void onTrackingFinished();
    void onViewCharts();
    void updateActions();
    void onImageChanged(int index, int total);

private:
    /// The inference entry points that run from a GUI slot. They share one
    /// asynchronous path: the inference itself must never run on the GUI
    /// thread (that is the class of defect this plugin used to carry — the
    /// window froze for the whole ONNX call), and only one may be in flight.
    enum class InferenceKind { Detection, Segmentation };

    void loadModelsInBackground();
    void setupMenus();
    void setupRibbonUI();
    void setupConnections();
    void requestInference(InferenceKind kind);
    void finishInference(InferenceKind kind, const QString& imagePath,
                         const cv::Mat& result);
    QFuture<cv::Mat> dispatchInference(std::function<cv::Mat()> task);
    static const char *inferenceStage(InferenceKind kind);

    IAppContext*   m_ctx         = nullptr;
    IAIService*    m_aiSvc       = nullptr;  ///< Looked up from ServiceRegistry
    IAIModelLoadService* m_modelLoader = nullptr; ///< AppCore-owned async loader
    /// AppCore-owned prediction writer. Resolved once in initialize() so the
    /// registry is not consulted on every detection click; null when the
    /// service is not registered (the plugin still runs, it just logs nothing).
    IAIPredictionLogService* m_predictionLog = nullptr;
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
    /// Set by onTrackingError() so onTrackingFinished() does not report a
    /// failed run as a completed one (the thread emits both, in order).
    bool m_trackingFailed = false;
    QString m_agentVideoPath;

    /// The async inference state. `m_inferenceRunning` is set and cleared on
    /// the GUI thread only; the task itself runs through blocking::run() on a
    /// thread the plugin owns. `m_inferenceTasks` lets cleanup() join that
    /// thread (bounded) before the plugin's code and services go away.
    bool m_inferenceRunning = false;
    bool m_cleanedUp = false;
    blocking::TaskTracker m_inferenceTasks;
};

#endif // AI_PROCESSOR_PLUGIN_H
