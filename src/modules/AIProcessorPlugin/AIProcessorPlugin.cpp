#include "AIProcessorPlugin.h"
#include "IAppContext.h"
#include "IShellService.h"
#include "ISettingsService.h"
#include "IViewerService.h"
#include "ISceneService.h"
#include "IAIService.h"
#include "IAIModelLoadService.h"
#include "SignalBus.h"
#include "AITrainDockWidget.h"
#include "AIProcessorRibbonUI.h"
#include "AIPredictionLogger.h"
#include "AppConstants.h"
#include "VideoTrackerThread.h"
#include "IconFactory.h"
#include "ModernMessageBox.h"
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QFileDialog>
#include <QMenu>
#include <QProcess>
#include <QUrl>
#include <QFile>
#include <QDockWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTextBrowser>
#include <QApplication>
#include <QTimer>
#include <QTcpSocket>
#include <QGroupBox>
#include <QStyle>
#include "IAppPaths.h"
#include <opencv2/opencv.hpp>
#include "IProgressDialog.h"

bool AIProcessorPlugin::initialize(IAppContext *context) {
  m_ctx = context;

  // Mandatory services: their absence is a composition-root wiring bug, not
  // a runtime condition. getRequired() aborts with the missing key instead of
  // letting a null pointer propagate into menu handlers.
  m_aiSvc = m_ctx->services()->getRequired<IAIService>();
  m_modelLoader = m_ctx->services()->getRequired<IAIModelLoadService>();
  {
      connect(m_modelLoader, &IAIModelLoadService::stageChanged, this,
              [this](const QString& stage) {
          if (!m_progressDialog) return;
          QString label;
          if (stage == QLatin1String("detection"))
              label = m_ctx->translate("aiproc.loading_det");
          else if (stage == QLatin1String("segmentation"))
              label = m_ctx->translate("aiproc.loading_seg");
          else if (stage == QLatin1String("tracking"))
              label = m_ctx->translate("aiproc.loading_track");
          if (label.isEmpty())
              label = QStringLiteral("Loading %1...").arg(stage);
          m_progressDialog->setLabelText(label);
      });
      connect(m_modelLoader, &IAIModelLoadService::finished, this,
              [this](bool success, const QString& error) {
          m_modelsLoading = false;
          if (m_progressDialog) m_progressDialog->hide();
          if (success) {
              qDebug() << "[AIProcessorPlugin] Background model loading completed.";
              if (m_aiSvc && m_aiSvc->isDetectionReady())
                  emit m_ctx->signalBus()->aiModelLoaded("detection");
              if (m_aiSvc && m_aiSvc->isSegmentationReady())
                  emit m_ctx->signalBus()->aiModelLoaded("segmentation");
              if (m_aiSvc && m_aiSvc->isTrackingReady())
                  emit m_ctx->signalBus()->aiModelLoaded("tracking");
          } else if (!error.isEmpty()) {
              qWarning().noquote() << "[AIProcessorPlugin]" << error;
          }
          updateActions();
      });
      connect(m_modelLoader, &IAIModelLoadService::cancelled, this, [this]() {
          m_modelsLoading = false;
          if (m_progressDialog) m_progressDialog->hide();
          qDebug() << "[AIProcessorPlugin] Model loading cancelled.";
          updateActions();
      });
  }

  m_progressDialog = m_ctx->shell()->getProgressDialog();
  if (m_progressDialog) {
      connect(m_progressDialog, &IProgressDialog::stopRequested, this, [this]() {
          if (m_modelLoader) m_modelLoader->cancel();
      });
  }
  
  // Setup Dock Widget
  m_trainDock = new AITrainDockWidget(m_ctx, m_ctx->shell()->dialogParent());
  m_ctx->shell()->addDockWidget(Qt::RightDockWidgetArea, m_trainDock);

  loadModelsInBackground();
  setupMenus();
  setupRibbonUI();
  setupConnections();

  updateActions();
  return true;
}

void AIProcessorPlugin::loadModelsInBackground() {
  if (!m_aiSvc || !m_modelLoader || m_modelsLoading || m_modelLoader->isLoading())
      return;

  auto* paths = m_ctx->services()->getRequired<IAppPaths>();
  if (!paths) return;

  const QString modelsPath = paths->modelsDir();
  const QString detPath = modelsPath + "/" + AppConstants::AIProcessor::detectionModelFile();
  const QString segPath = modelsPath + "/" + AppConstants::AIProcessor::segmentationModelFile();
  const QString trackPath = modelsPath + "/" + AppConstants::AIProcessor::trackingModelFile();

  if (!QFile::exists(detPath) && !QFile::exists(segPath) && !QFile::exists(trackPath))
      return;

  m_modelsLoading = true;
  if (m_runDetAct) m_runDetAct->setEnabled(false);
  if (m_runSegAct) m_runSegAct->setEnabled(false);

  QString firstLabel = m_ctx->translate("aiproc.loading_det");
  if (firstLabel.isEmpty()) firstLabel = "Loading Detection...";
  if (m_progressDialog) {
      m_progressDialog->setLabelText(firstLabel);
      m_progressDialog->setRange(0, 0);
      m_progressDialog->show();
      m_progressDialog->centerOnWidget(m_ctx->shell()->dialogParent());
  }

  // The worker lives in AppCore (IAIModelLoadService). Keeping it out of this
  // plugin DLL is what makes plugin unload safe while a model is loading.
  m_modelLoader->loadModelsAsync(detPath, segPath, trackPath);
}

void AIProcessorPlugin::setupMenus() {
  QMenu *aiMenu = m_ctx->shell()->getMenu(AppConstants::MenuIds::AiProcessor);
  aiMenu->clear();

  m_detMenu = aiMenu->addMenu(IconFactory::createModern("🔍", QColor("#f59e0b"), QColor("#d97706")), m_ctx->translate("ai.detection"));
  m_runDetAct = m_detMenu->addAction(IconFactory::createModern("🔍", QColor("#f59e0b"), QColor("#d97706")), m_ctx->translate("ai.run_detection"), this, &AIProcessorPlugin::onObjectDetection);
  m_hideDetAct = m_detMenu->addAction(IconFactory::createModern("👁️", QColor("#6b7280"), QColor("#4b5563")), m_ctx->translate("ai.hide_results"), this, &AIProcessorPlugin::onHideAIResults);

  m_segMenu = aiMenu->addMenu(IconFactory::createModern("🎯", QColor("#10b981"), QColor("#059669")), m_ctx->translate("ai.segmentation"));
  m_runSegAct = m_segMenu->addAction(IconFactory::createModern("🎯", QColor("#10b981"), QColor("#059669")), m_ctx->translate("ai.run_segmentation"), this, &AIProcessorPlugin::onSegmentation);
  m_hideSegAct = m_segMenu->addAction(IconFactory::createModern("👁️", QColor("#6b7280"), QColor("#4b5563")), m_ctx->translate("ai.hide_results"), this, &AIProcessorPlugin::onHideAIResults);

  m_trackMenu = aiMenu->addMenu(IconFactory::createModern("🎥", QColor("#ef4444"), QColor("#b91c1c")), m_ctx->translate("ai.video_tracking"));
  m_runTrackingAct = m_trackMenu->addAction(IconFactory::createModern("🎥", QColor("#ef4444"), QColor("#b91c1c")), m_ctx->translate("ai.run_video_tracking"), this, &AIProcessorPlugin::onObjectTracking);
  m_pauseTrackingAct = m_trackMenu->addAction(IconFactory::createModern("⏸️", QColor("#f59e0b"), QColor("#d97706")), m_ctx->translate("ai.pause_resume"), this, &AIProcessorPlugin::onPauseResumeTracking);
  m_stopTrackingAct = m_trackMenu->addAction(IconFactory::createModern("⏹️", QColor("#ef4444"), QColor("#b91c1c")), m_ctx->translate("ai.stop_tracking"), this, &AIProcessorPlugin::onStopTracking);

  aiMenu->addSeparator();
  aiMenu->addAction(IconFactory::createModern("⚙️", QColor("#8b5cf6"), QColor("#7c3aed")), m_ctx->translate("ai.training"), this, &AIProcessorPlugin::onTrainModel);
  aiMenu->addAction(IconFactory::createModern("📊", QColor("#06b6d4"), QColor("#0891b2")), m_ctx->translate("ai.charts"), this, &AIProcessorPlugin::onViewCharts);
}

void AIProcessorPlugin::setupRibbonUI() {
  if (QWidget* panel = m_ctx->shell()->getTabPanel(AppConstants::TabIds::AiProcessor)) {
      m_ribbonUI = new AIProcessorRibbonUI(m_ctx, panel, this);
      connect(m_ribbonUI->btnDet(), &QToolButton::clicked, this, &AIProcessorPlugin::onObjectDetection);
      connect(m_ribbonUI->btnSeg(), &QToolButton::clicked, this, &AIProcessorPlugin::onSegmentation);
      connect(m_ribbonUI->btnTrack(), &QToolButton::clicked, this, &AIProcessorPlugin::onObjectTracking);
      connect(m_ribbonUI->btnPauseTrack(), &QToolButton::clicked, this, &AIProcessorPlugin::onPauseResumeTracking);
      connect(m_ribbonUI->btnStopTrack(), &QToolButton::clicked, this, &AIProcessorPlugin::onStopTracking);
      connect(m_ribbonUI->btnHide(), &QToolButton::clicked, this, &AIProcessorPlugin::onHideAIResults);
      connect(m_ribbonUI->btnTrain(), &QToolButton::clicked, this, &AIProcessorPlugin::onTrainModel);
      connect(m_ribbonUI->btnChart(), &QToolButton::clicked, this, &AIProcessorPlugin::onViewCharts);
  }
}

void AIProcessorPlugin::setupConnections() {
  connect(m_ctx->signalBus(), &SignalBus::stateChanged, this, &AIProcessorPlugin::updateActions);
  connect(m_ctx->signalBus(), &SignalBus::languageChanged, this, &AIProcessorPlugin::updateActions);
  connect(m_ctx->signalBus(), &SignalBus::imageIndexChanged, this, &AIProcessorPlugin::onImageChanged);
  connect(m_ctx->signalBus(), &SignalBus::agentUiActionRequested, this,
          [this](const QString &action, const QVariantMap &parameters) {
      // "path" is filled by AgentActionManifest::applySampleDefaults() from
      // Config/agent_action_manifest.json when the agent provides none.
      const QString path = parameters.value("path").toString();
      bool handled = false;
      if (action == "ai.run_detection" || action == "ai.run_segmentation") {
          if (!path.isEmpty()) {
              m_ctx->viewer()->setCurrent2DImagePath(path);
              if (action == "ai.run_detection") onObjectDetection();
              else onSegmentation();
              handled = true;
          }
      } else if (action == "ai.video_tracking") {
          if (!path.isEmpty()) {
              m_agentVideoPath = path;
              onObjectTracking();
              handled = true;
          }
      } else if (action == "ai.hide_results") {
          onHideAIResults();
          handled = true;
      } else if (action == "ai.training_model") {
          if (m_trainDock) m_trainDock->startTraining(true);
          handled = m_trainDock != nullptr;
      } else if (action == "ai.view_training_charts") {
          onViewCharts();
          handled = true;
      }
      const QString requestId = parameters.value("request_id").toString();
      if (!requestId.isEmpty() && action.startsWith("ai."))
          emit m_ctx->signalBus()->agentUiActionCompleted(requestId, handled,
              QVariantMap{{"action", action}, {"error", handled ? "" : "Required AI input is unavailable"}});
  });
}

void AIProcessorPlugin::cleanup() {
    // The model-load worker belongs to AppCore and may still be running; the
    // plugin only stops observing it. This avoids executing plugin code after
    // QPluginLoader unloads this DLL.
    if (m_modelLoader) {
        disconnect(m_modelLoader, nullptr, this, nullptr);
        m_modelLoader->cancel();
    }
    m_modelsLoading = false;

    if (m_tensorboardWaitTimer) {
        m_tensorboardWaitTimer->stop();
    }
    if (m_tensorboardProcess && m_tensorboardProcess->state() != QProcess::NotRunning) {
        m_tensorboardProcess->terminate();
        if (!m_tensorboardProcess->waitForFinished(2000)) {
            m_tensorboardProcess->kill();
        }
    }
    if (m_trackerThread) {
        m_trackerThread->stopAndWait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }

    // AITrainDockWidget is a plugin-defined QWidget parented to MainWindow.
    // Delete it synchronously before QPluginLoader unloads this DLL.
    //
    // It must be deleted directly: IShellService::deleteDockWidget() takes a
    // QDockWidget*, and handing it this plain QWidget made the shell delete
    // it through the wrong vtable — the heap was corrupted and the process
    // died on exit (caught by TestPluginSmoke's initialize/cleanup cycle).
    if (m_trainDock) {
        delete m_trainDock;
        m_trainDock = nullptr;
    }
    delete m_ribbonUI;
    m_ribbonUI = nullptr;
}

void AIProcessorPlugin::onTrainModel() {
    if (m_trainDock) {
        m_trainDock->startTraining();
    }
}

void AIProcessorPlugin::onObjectDetection() {
  QString currentImg = m_ctx->viewer()->getCurrent2DImagePath();
  if (currentImg.isEmpty()) {
      ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.warning"), m_ctx->translate("aiproc.load_img_warn"));
      return;
  }
  
  // Clear any active 3D models before showing 2D results
  m_ctx->scene()->clear3DModel();
  m_ctx->scene()->clearPointCloud();
  m_ctx->scene()->resetToSingleRenderer();

  if (!m_aiSvc || !m_aiSvc->isDetectionReady()) {
      ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"), m_ctx->translate("aiproc.model_not_loaded_warn"));
      return;
  }
  emit m_ctx->signalBus()->aiInferenceStarted("detection");
  cv::Mat res = m_aiSvc->runDetection(cv::imread(currentImg.toStdString()));
  emit m_ctx->signalBus()->aiInferenceFinished("detection");

  m_ctx->viewer()->setAIMode(AIMode::Detection);
  m_ctx->scene()->showImage(res);
  m_ctx->scene()->render();
  m_ctx->shell()->updateMenuStates();

  AIPredictionLogger::logAsync(m_ctx->services()->getRequired<IAppPaths>(), "detection", currentImg, res);
}

void AIProcessorPlugin::onSegmentation() {
  QString currentImg = m_ctx->viewer()->getCurrent2DImagePath();
  if (currentImg.isEmpty()) {
      ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.warning"), m_ctx->translate("aiproc.load_img_warn"));
      return;
  }

  // Clear any active 3D models before showing 2D results
  m_ctx->scene()->clear3DModel();
  m_ctx->scene()->clearPointCloud();
  m_ctx->scene()->resetToSingleRenderer();

  if (!m_aiSvc || !m_aiSvc->isSegmentationReady()) {
      ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"), m_ctx->translate("aiproc.model_not_loaded_warn"));
      return;
  }
  emit m_ctx->signalBus()->aiInferenceStarted("segmentation");
  cv::Mat res = m_aiSvc->runSegmentation(cv::imread(currentImg.toStdString()));
  emit m_ctx->signalBus()->aiInferenceFinished("segmentation");

  m_ctx->viewer()->setAIMode(AIMode::Segmentation);
  m_ctx->scene()->showImage(res);
  m_ctx->scene()->render();
  m_ctx->shell()->updateMenuStates();

  AIPredictionLogger::logAsync(m_ctx->services()->getRequired<IAppPaths>(), "segmentation", currentImg, res);
}

void AIProcessorPlugin::onHideAIResults() {
    const QString originalPath = m_ctx->viewer()->getCurrent2DImagePath();
    m_ctx->viewer()->setAIMode(AIMode::None);
    if (!originalPath.isEmpty() && QFile::exists(originalPath)) {
        m_ctx->scene()->showImage(originalPath);
        m_ctx->scene()->render();
    } else {
        m_ctx->viewer()->loadCurrentIndexImage();
    }
    m_ctx->shell()->updateMenuStates();
}

void AIProcessorPlugin::onViewCharts() {
  auto* paths = m_ctx->services()->getRequired<IAppPaths>();
  if (!paths) return;
  QString aiComputerVisionDir = paths->aiComputerVisionDir();
  QString runsDir = QDir::cleanPath(aiComputerVisionDir + "/runs");
  if (!QDir(runsDir).exists()) {
      ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"),
                                m_ctx->translate("aiproc.tb_dir_missing").arg(runsDir));
      return;
  }

  // TensorBoard is launched with `python -m ...`, so killing tensorboard.exe did
  // not stop prior instances.  Reuse an already listening local server instead.
  QTcpSocket socket;
  socket.connectToHost(QHostAddress::LocalHost, AppConstants::AIProcessor::TENSORBOARD_PORT);
  if (socket.waitForConnected(250)) {
      QDesktopServices::openUrl(QUrl(AppConstants::AIProcessor::tensorboardUrl()));
      return;
  }

  if (m_tensorboardProcess && m_tensorboardProcess->state() != QProcess::NotRunning) {
      return;
  }

  if (m_tensorboardWaitTimer) {
      m_tensorboardWaitTimer->stop();
      m_tensorboardWaitTimer->deleteLater();
      m_tensorboardWaitTimer = nullptr;
  }

  m_tensorboardProcess = new QProcess(this);
  m_tensorboardProcess->setWorkingDirectory(aiComputerVisionDir);
  m_tensorboardProcess->setProcessChannelMode(QProcess::MergedChannels);
  m_tensorboardProcess->start("python", {"-m", "tensorboard.main", "--logdir", runsDir,
                                            "--port", QString::number(AppConstants::AIProcessor::TENSORBOARD_PORT)});

  if (!m_progressDialog) return;
  const int totalSeconds = AppConstants::AIProcessor::TENSORBOARD_WAIT_SECONDS;
  m_progressDialog->setLabelText(m_ctx->translate("aiproc.tb_seconds").arg(totalSeconds));
  m_progressDialog->setValue(0);
  m_progressDialog->show();
  m_progressDialog->centerOnWidget();

  QProcess *process = m_tensorboardProcess;
  QTimer *waitTimer = new QTimer(this);
  m_tensorboardWaitTimer = waitTimer;
  waitTimer->setInterval(250);
  auto elapsedMs = std::make_shared<int>(0);
  connect(waitTimer, &QTimer::timeout, this, [this, process, waitTimer, elapsedMs, totalSeconds]() {
      *elapsedMs += 250;
      QTcpSocket check;
      check.connectToHost(QHostAddress::LocalHost, AppConstants::AIProcessor::TENSORBOARD_PORT);
      if (check.waitForConnected(100)) {
          waitTimer->stop();
          waitTimer->deleteLater();
          if (m_tensorboardWaitTimer == waitTimer) m_tensorboardWaitTimer = nullptr;
          m_progressDialog->hide();
          m_progressDialog->reset();
          QDesktopServices::openUrl(QUrl(AppConstants::AIProcessor::tensorboardUrl()));
      } else if (*elapsedMs >= totalSeconds * 1000) {
          waitTimer->stop();
          waitTimer->deleteLater();
          if (m_tensorboardWaitTimer == waitTimer) m_tensorboardWaitTimer = nullptr;
          const QString output = QString::fromLocal8Bit(process->readAll());
          m_progressDialog->hide();
          ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"),
              m_ctx->translate("aiproc.tb_start_failed") + "\n" + output);
      } else if (m_progressDialog) {
          m_progressDialog->setValue((*elapsedMs * 100) / (totalSeconds * 1000));
          int remainingSeconds = totalSeconds - (*elapsedMs / 1000);
          m_progressDialog->setLabelText(m_ctx->translate("aiproc.tb_seconds").arg(remainingSeconds));
      }
  });
  connect(m_progressDialog, &IProgressDialog::stopRequested, waitTimer, [this, waitTimer]() {
      waitTimer->stop();
      waitTimer->deleteLater();
      if (m_tensorboardWaitTimer == waitTimer) m_tensorboardWaitTimer = nullptr;
      m_progressDialog->hide();
  });
  connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
          [this, process, waitTimer](int, QProcess::ExitStatus) {
      if (m_tensorboardWaitTimer != waitTimer) return;
      waitTimer->stop();
      waitTimer->deleteLater();
      if (m_tensorboardWaitTimer == waitTimer) m_tensorboardWaitTimer = nullptr;
      m_progressDialog->hide();
      ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"),
          m_ctx->translate("aiproc.tb_start_failed") + "\n" +
          QString::fromLocal8Bit(process->readAll()));
  });
  waitTimer->start();
}

void AIProcessorPlugin::onImageChanged(int index, int total) {
    Q_UNUSED(index);
    Q_UNUSED(total);
    AIMode mode = m_ctx->viewer()->getCurrentAIMode();
    if (mode == AIMode::Detection) {
        onObjectDetection();
    } else if (mode == AIMode::Segmentation) {
        onSegmentation();
    }
}

void AIProcessorPlugin::onObjectTracking() {
    if (!m_aiSvc || !m_aiSvc->isTrackingReady()) {
        ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"), m_ctx->translate("aiproc.model_not_loaded_warn"));
        return;
    }

    QString videoPath = m_agentVideoPath;
    m_agentVideoPath.clear();
    if (videoPath.isEmpty()) {
        QString lastUsedPath = m_ctx->settings()->getLastUsedPath("ai_video_tracking");
        videoPath = QFileDialog::getOpenFileName(m_ctx->shell()->dialogParent(), m_ctx->translate("aiproc.select_video_tracking"), lastUsedPath, "Video Files (*.mp4 *.avi *.mkv)");
    }
    if (videoPath.isEmpty()) return;
    
    m_ctx->settings()->setLastUsedPath("ai_video_tracking", QFileInfo(videoPath).absolutePath());

    // Clear current state
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->resetToSingleRenderer();
    m_ctx->viewer()->setAIMode(AIMode::Tracking);
    
    // Stop existing tracking if any
    if (m_trackerThread) {
        m_trackerThread->stopAndWait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }

    m_aiSvc->resetTrackingState();

    m_trackerThread = new VideoTrackerThread(m_aiSvc, videoPath, this);
    connect(m_trackerThread, &VideoTrackerThread::frameProcessed, this, &AIProcessorPlugin::onTrackingFrameReceived);
    connect(m_trackerThread, &VideoTrackerThread::finishedTracking, this, &AIProcessorPlugin::onTrackingFinished);
    
    m_trackingPaused = false;
    m_trackerThread->start();

    updateActions();
}

void AIProcessorPlugin::onPauseResumeTracking() {
    if (!m_trackerThread) return;
    
    m_trackingPaused = !m_trackingPaused;
    if (m_trackingPaused) {
        m_trackerThread->pause();
        if (m_pauseTrackingAct) m_pauseTrackingAct->setText(m_ctx->translate("ai.resume"));
        if (m_ribbonUI) m_ribbonUI->btnPauseTrack()->setText(m_ctx->translate("ai.resume"));
    } else {
        m_trackerThread->resume();
        if (m_pauseTrackingAct) m_pauseTrackingAct->setText(m_ctx->translate("ai.pause"));
        if (m_ribbonUI) m_ribbonUI->btnPauseTrack()->setText(m_ctx->translate("ai.pause"));
    }
}

void AIProcessorPlugin::onStopTracking() {
    if (m_trackerThread) {
        m_trackerThread->stopAndWait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }
    m_ctx->viewer()->setAIMode(AIMode::None);
    updateActions();
}

void AIProcessorPlugin::onTrackingFrameReceived(const cv::Mat& frame) {
    if (m_ctx->viewer()->getCurrentAIMode() != AIMode::Tracking) return;
    m_ctx->scene()->showImage(frame);
    m_ctx->scene()->render();
}

void AIProcessorPlugin::onTrackingFinished() {
    if (m_trackerThread) {
        m_trackerThread->stopAndWait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }
    m_ctx->viewer()->setAIMode(AIMode::None);
    ModernMessageBox::information(m_ctx->shell()->dialogParent(), m_ctx->translate("aiproc.track_finished"), m_ctx->translate("aiproc.track_finished_msg"));
    updateActions();
}

void AIProcessorPlugin::updateActions() {
    QMenu *aiMenu = m_ctx->shell()->getMenu(AppConstants::MenuIds::AiProcessor);
    if (aiMenu) {
        aiMenu->setTitle(m_ctx->translate("ai.menu"));
    }
    if (!m_runDetAct) return;
  AIMode mode = m_ctx->viewer()->getCurrentAIMode();
  bool has2D = !m_ctx->viewer()->getCurrent2DImagePath().isEmpty();

  bool detReady = m_aiSvc && m_aiSvc->isDetectionReady();
  bool segReady = m_aiSvc && m_aiSvc->isSegmentationReady();
  bool trackReady = m_aiSvc && m_aiSvc->isTrackingReady();
  bool isTrackingMode = (mode == AIMode::Tracking);

  // Re-apply translated menu/action texts (also runs on languageChanged).
  if (m_detMenu)   m_detMenu->setTitle(m_ctx->translate("ai.detection"));
  if (m_segMenu)   m_segMenu->setTitle(m_ctx->translate("ai.segmentation"));
  if (m_trackMenu) m_trackMenu->setTitle(m_ctx->translate("ai.video_tracking"));
  m_runDetAct->setText(m_ctx->translate("ai.run_detection"));
  m_hideDetAct->setText(m_ctx->translate("ai.hide_results"));
  m_runSegAct->setText(m_ctx->translate("ai.run_segmentation"));
  m_hideSegAct->setText(m_ctx->translate("ai.hide_results"));
  m_runTrackingAct->setText(m_ctx->translate("ai.run_video_tracking"));
  m_pauseTrackingAct->setText(m_ctx->translate(
      isTrackingMode ? (m_trackingPaused ? "ai.resume" : "ai.pause") : "ai.pause_resume"));
  m_stopTrackingAct->setText(m_ctx->translate("ai.stop_tracking"));

  if (m_modelsLoading) {
      m_runDetAct->setEnabled(false);
      m_runSegAct->setEnabled(false);
      m_runTrackingAct->setEnabled(false);
      m_pauseTrackingAct->setEnabled(false);
      m_stopTrackingAct->setEnabled(false);
      if (m_ribbonUI) {
          m_ribbonUI->btnDet()->setText(m_ctx->translate("ai.run_detection"));
          m_ribbonUI->btnDet()->setEnabled(false);
          m_ribbonUI->btnSeg()->setText(m_ctx->translate("ai.run_segmentation"));
          m_ribbonUI->btnSeg()->setEnabled(false);
          m_ribbonUI->btnTrack()->setEnabled(false);
          m_ribbonUI->btnPauseTrack()->setEnabled(false);
          m_ribbonUI->btnStopTrack()->setEnabled(false);
          m_ribbonUI->btnPauseTrack()->setVisible(false);
          m_ribbonUI->btnStopTrack()->setVisible(false);
          m_ribbonUI->btnTrack()->setVisible(true);
      }
  } else {
      m_runDetAct->setEnabled(!isTrackingMode && mode != AIMode::Detection && has2D && detReady);
      m_runSegAct->setEnabled(!isTrackingMode && mode != AIMode::Segmentation && has2D && segReady);
      m_runTrackingAct->setEnabled(!isTrackingMode && trackReady);
      m_pauseTrackingAct->setEnabled(isTrackingMode);
      m_stopTrackingAct->setEnabled(isTrackingMode);
      
      if (m_ribbonUI) {
          m_ribbonUI->btnDet()->setText(m_ctx->translate("ai.run_detection"));
          m_ribbonUI->btnDet()->setEnabled(!isTrackingMode && mode != AIMode::Detection && has2D && detReady);
          m_ribbonUI->btnSeg()->setText(m_ctx->translate("ai.run_segmentation"));
          m_ribbonUI->btnSeg()->setEnabled(!isTrackingMode && mode != AIMode::Segmentation && has2D && segReady);
          m_ribbonUI->btnTrack()->setEnabled(!isTrackingMode && trackReady);
          
          m_ribbonUI->btnPauseTrack()->setEnabled(isTrackingMode);
          m_ribbonUI->btnStopTrack()->setEnabled(isTrackingMode);
          
          m_ribbonUI->btnTrack()->setVisible(!isTrackingMode);
          m_ribbonUI->btnPauseTrack()->setVisible(isTrackingMode);
          m_ribbonUI->btnStopTrack()->setVisible(isTrackingMode);
      }
  }

  m_hideDetAct->setEnabled(mode == AIMode::Detection);
  m_hideSegAct->setEnabled(mode == AIMode::Segmentation);

  if (m_trainDock) m_trainDock->retranslate();

  if (m_ribbonUI) {
      m_ribbonUI->btnHide()->setText(m_ctx->translate("ai.hide_results"));
      m_ribbonUI->btnHide()->setEnabled(mode != AIMode::None);
      m_ribbonUI->btnTrack()->setText(m_ctx->translate("ai.video_tracking"));
      m_ribbonUI->btnTrain()->setText(m_ctx->translate("ai.training"));
      m_ribbonUI->btnChart()->setText(m_ctx->translate("ai.charts"));
      m_ribbonUI->btnPauseTrack()->setText(m_ctx->translate(m_trackingPaused ? "ai.resume" : "ai.pause"));
      m_ribbonUI->btnStopTrack()->setText(m_ctx->translate("aiproc.stop"));
      
      // Update GroupBox titles
      if (QLabel *lbl = m_ribbonUI->groupAI()->findChild<QLabel*>("groupTitleLabel")) {
          lbl->setText(m_ctx->translate("ai.menu"));
      }
      if (QLabel *lbl = m_ribbonUI->groupTrain()->findChild<QLabel*>("groupTitleLabel")) {
          lbl->setText(m_ctx->translate("ai.training"));
      }
  }
}
