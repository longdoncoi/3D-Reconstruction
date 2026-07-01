#include "AIProcessorPlugin.h"
#include "IAppContext.h"
#include "ISettingsService.h"
#include "IViewerService.h"
#include "ISceneService.h"
#include "IAIService.h"
#include "SignalBus.h"
#include "Image2DLoader.h"
#include "AITrainDockWidget.h"
#include "AIProcessorRibbonUI.h"
#include "LanguageManager.h"
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
#include <QDateTime>
#include <QFile>
#include <QDockWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTextBrowser>
#include <QApplication>
#include <QVTKOpenGLNativeWidget.h>
#include <QTimer>
#include <QtConcurrent>
#include <QThreadPool>
#include <QGroupBox>
#include <QStyle>
#include "AppConfig.h"
#include <vtkRenderWindow.h>
#include <opencv2/opencv.hpp>
#include "../../utils/CustomProgressDialog.h"

void AIProcessorPlugin::initialize(IAppContext *context) {
  m_ctx = context;

  // Look up IAIService
  m_aiSvc = m_ctx->services()->get<IAIService>();
  if (!m_aiSvc) {
      qWarning() << "[AIProcessorPlugin] IAIService not found in ServiceRegistry!";
  }

  m_progressDialog = new CustomProgressDialog(m_ctx->mainWindow());
  
  // Setup Dock Widget
  m_trainDock = new AITrainDockWidget(m_ctx, m_ctx->mainWindow());
  m_ctx->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, m_trainDock);

  loadModelsInBackground();
  setupMenus();
  setupRibbonUI();
  setupConnections();

  updateActions();
}

void AIProcessorPlugin::loadModelsInBackground() {
  if (!m_aiSvc) return;
  QString modelsPath = AppConfig::instance().modelsDir();
  QString detPath = modelsPath + "/yolo11n.onnx";
  QString segPath = modelsPath + "/yolo11n-seg.onnx";
  QString trackPath = modelsPath + "/yolo11x-tracking.onnx";
  bool detExists = QFile::exists(detPath);
  bool segExists = QFile::exists(segPath);
  bool trackExists = QFile::exists(trackPath);

  if (detExists || segExists || trackExists) {
      m_modelsLoading = true;
      m_cancelModelLoad.store(false);

      if (m_runDetAct) m_runDetAct->setEnabled(false);
      if (m_runSegAct) m_runSegAct->setEnabled(false);

      m_modelLoadWatcher = new QFutureWatcher<void>(this);
      connect(m_modelLoadWatcher, &QFutureWatcher<void>::finished, this, [this]() {
          m_modelsLoading = false;
          if (m_progressDialog) m_progressDialog->hide();
          if (m_modelLoadStopConn) {
              disconnect(m_modelLoadStopConn);
              m_modelLoadStopConn = {};
          }
          if (m_cancelModelLoad.load()) {
              qDebug() << "[AIProcessorPlugin] Model loading cancelled.";
          } else {
              qDebug() << "[AIProcessorPlugin] Background model loading completed.";
              if (m_aiSvc && m_aiSvc->isDetectionReady())
                  emit m_ctx->signalBus()->aiModelLoaded("detection");
              if (m_aiSvc && m_aiSvc->isSegmentationReady())
                  emit m_ctx->signalBus()->aiModelLoaded("segmentation");
              if (m_aiSvc && m_aiSvc->isTrackingReady())
                  emit m_ctx->signalBus()->aiModelLoaded("tracking");
          }
          updateActions(); 
      });

      if (m_progressDialog) {
          m_modelLoadStopConn = connect(m_progressDialog, &CustomProgressDialog::stopRequested, this, [this]() {
              m_cancelModelLoad.store(true);
              m_modelsLoading = false;
              if (m_progressDialog) m_progressDialog->hide();
              updateActions();
          });

          QString firstLabel = (detExists) ? m_ctx->translate("aiproc.loading_det") : m_ctx->translate("aiproc.loading_seg");
          if (firstLabel.isEmpty()) firstLabel = (detExists) ? "Loading Detection..." : "Loading Segmentation...";
          m_progressDialog->setLabelText(firstLabel);
          m_progressDialog->setRange(0, 0);
          m_progressDialog->show();
          m_progressDialog->centerOnWidget(m_ctx->mainWindow());
      }

      IAIService* svc   = m_aiSvc;
      auto *dlg         = m_progressDialog;
      auto *ctx         = m_ctx;
      bool _detExists   = detExists;
      bool _segExists   = segExists;
      bool _trackExists = trackExists;
      auto *cancelFlag  = &m_cancelModelLoad;

      QFuture<void> future = QtConcurrent::run([svc, dlg, ctx, detPath, segPath, trackPath, _detExists, _segExists, _trackExists, cancelFlag]() {
          if (_detExists && !cancelFlag->load()) {
              QString detMsg = ctx->translate("aiproc.loading_det");
              if (detMsg.isEmpty()) detMsg = "Loading Detection...";
              QMetaObject::invokeMethod(dlg, [dlg, detMsg]() { if (dlg) dlg->setLabelText(detMsg); }, Qt::QueuedConnection);
              svc->loadDetectionModel(detPath);
          }
          if (_segExists && !cancelFlag->load()) {
              QString segMsg = ctx->translate("aiproc.loading_seg");
              if (segMsg.isEmpty()) segMsg = "Loading Segmentation...";
              QMetaObject::invokeMethod(dlg, [dlg, segMsg]() { if (dlg) dlg->setLabelText(segMsg); }, Qt::QueuedConnection);
              svc->loadSegmentationModel(segPath);
          }
          if (_trackExists && !cancelFlag->load()) {
              QString trackMsg = ctx->translate("aiproc.loading_track");
              if (trackMsg.isEmpty()) trackMsg = "Loading Tracking...";
              QMetaObject::invokeMethod(dlg, [dlg, trackMsg]() { if (dlg) dlg->setLabelText(trackMsg); }, Qt::QueuedConnection);
              svc->loadTrackingModel(trackPath);
          }
      });
      m_modelLoadWatcher->setFuture(future);
  }
}

void AIProcessorPlugin::setupMenus() {
  QMenu *aiMenu = m_ctx->getMenu("ai_menu");
  aiMenu->clear();

  QMenu *detMenu = aiMenu->addMenu(IconFactory::createModern("🔍", QColor("#f59e0b"), QColor("#d97706")), "Object Detection");
  m_runDetAct = detMenu->addAction(IconFactory::createModern("🔍", QColor("#f59e0b"), QColor("#d97706")), m_ctx->translate("ai.run_detection"), this, &AIProcessorPlugin::onObjectDetection);
  m_hideDetAct = detMenu->addAction(IconFactory::createModern("👁️", QColor("#6b7280"), QColor("#4b5563")), m_ctx->translate("ai.hide_results"), this, &AIProcessorPlugin::onHideAIResults);

  QMenu *segMenu = aiMenu->addMenu(IconFactory::createModern("🎯", QColor("#10b981"), QColor("#059669")), "Segmentation");
  m_runSegAct = segMenu->addAction(IconFactory::createModern("🎯", QColor("#10b981"), QColor("#059669")), m_ctx->translate("ai.run_segmentation"), this, &AIProcessorPlugin::onSegmentation);
  m_hideSegAct = segMenu->addAction(IconFactory::createModern("👁️", QColor("#6b7280"), QColor("#4b5563")), m_ctx->translate("ai.hide_results"), this, &AIProcessorPlugin::onHideAIResults);

  QMenu *trackMenu = aiMenu->addMenu(IconFactory::createModern("🎥", QColor("#ef4444"), QColor("#b91c1c")), "Video Tracking");
  m_runTrackingAct = trackMenu->addAction(IconFactory::createModern("🎥", QColor("#ef4444"), QColor("#b91c1c")), "Run Video Tracking", this, &AIProcessorPlugin::onObjectTracking);
  m_pauseTrackingAct = trackMenu->addAction(IconFactory::createModern("⏸️", QColor("#f59e0b"), QColor("#d97706")), "Pause/Resume", this, &AIProcessorPlugin::onPauseResumeTracking);
  m_stopTrackingAct = trackMenu->addAction(IconFactory::createModern("⏹️", QColor("#ef4444"), QColor("#b91c1c")), "Stop Tracking", this, &AIProcessorPlugin::onStopTracking);

  aiMenu->addSeparator();
  aiMenu->addAction(IconFactory::createModern("⚙️", QColor("#8b5cf6"), QColor("#7c3aed")), m_ctx->translate("ai.training"), this, &AIProcessorPlugin::onTrainModel);
  aiMenu->addAction(IconFactory::createModern("📊", QColor("#06b6d4"), QColor("#0891b2")), m_ctx->translate("ai.charts"), this, &AIProcessorPlugin::onViewCharts);
}

void AIProcessorPlugin::setupRibbonUI() {
  if (QWidget* panel = m_ctx->getTabPanel("tab.ai")) {
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
}

void AIProcessorPlugin::cleanup() {
    if (m_trackerThread) {
        m_trackerThread->stop();
        m_trackerThread->wait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }
}

void AIProcessorPlugin::onTrainModel() {
    if (m_trainDock) {
        m_trainDock->startTraining();
    }
}

void AIProcessorPlugin::onObjectDetection() {
  QString currentImg = m_ctx->viewer()->getCurrent2DImagePath();
  if (currentImg.isEmpty()) {
      ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.warning"), m_ctx->translate("aiproc.load_img_warn"));
      return;
  }
  
  // Clear any active 3D models before showing 2D results
  m_ctx->scene()->clear3DModel();
  m_ctx->scene()->clearPointCloud();
  m_ctx->scene()->resetToSingleRenderer();

  if (!m_aiSvc || !m_aiSvc->isDetectionReady()) {
      ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.error"), m_ctx->translate("aiproc.model_not_loaded_warn"));
      return;
  }
  emit m_ctx->signalBus()->aiInferenceStarted("detection");
  cv::Mat res = m_aiSvc->runDetection(cv::imread(currentImg.toStdString()));
  emit m_ctx->signalBus()->aiInferenceFinished("detection");

  m_ctx->viewer()->setAIMode(AIMode::Detection);
  m_ctx->scene()->setTextureActor(Image2DLoader::loadFromMat(res));
  m_ctx->scene()->vtkWidget()->renderWindow()->Render();
  m_ctx->updateMenuStates();

  // Log prediction image asynchronously
  QString currentDate = QDateTime::currentDateTime().toString("yyyy-MM-dd");
  QString predictDir = AppConfig::instance().predictDir("detection") + "/" + currentDate;
  QDir().mkpath(predictDir);
  QString logPath = predictDir + "/" + QFileInfo(currentImg).baseName() + "_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";
  
  cv::Mat resClone = res.clone();
  QThreadPool::globalInstance()->start([logPath, resClone]() {
      cv::imwrite(logPath.toStdString(), resClone);
  });
}

void AIProcessorPlugin::onSegmentation() {
  QString currentImg = m_ctx->viewer()->getCurrent2DImagePath();
  if (currentImg.isEmpty()) {
      ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.warning"), m_ctx->translate("aiproc.load_img_warn"));
      return;
  }

  // Clear any active 3D models before showing 2D results
  m_ctx->scene()->clear3DModel();
  m_ctx->scene()->clearPointCloud();
  m_ctx->scene()->resetToSingleRenderer();

  if (!m_aiSvc || !m_aiSvc->isSegmentationReady()) {
      ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.error"), m_ctx->translate("aiproc.model_not_loaded_warn"));
      return;
  }
  emit m_ctx->signalBus()->aiInferenceStarted("segmentation");
  cv::Mat res = m_aiSvc->runSegmentation(cv::imread(currentImg.toStdString()));
  emit m_ctx->signalBus()->aiInferenceFinished("segmentation");

  m_ctx->viewer()->setAIMode(AIMode::Segmentation);
  m_ctx->scene()->setTextureActor(Image2DLoader::loadFromMat(res));
  m_ctx->scene()->vtkWidget()->renderWindow()->Render();
  m_ctx->updateMenuStates();

  // Log prediction image asynchronously
  QString currentDate = QDateTime::currentDateTime().toString("yyyy-MM-dd");
  QString predictDir = AppConfig::instance().predictDir("segmentation") + "/" + currentDate;
  QDir().mkpath(predictDir);
  QString logPath = predictDir + "/" + QFileInfo(currentImg).baseName() + "_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";
  
  cv::Mat resClone = res.clone();
  QThreadPool::globalInstance()->start([logPath, resClone]() {
      cv::imwrite(logPath.toStdString(), resClone);
  });
}

void AIProcessorPlugin::onHideAIResults() {
    const QString originalPath = m_ctx->viewer()->getCurrent2DImagePath();
    m_ctx->viewer()->setAIMode(AIMode::None);
    if (!originalPath.isEmpty() && QFile::exists(originalPath)) {
        m_ctx->scene()->setTextureActor(Image2DLoader::load(originalPath));
        m_ctx->scene()->vtkWidget()->renderWindow()->Render();
    } else {
        m_ctx->viewer()->loadCurrentIndexImage();
    }
    m_ctx->updateMenuStates();
}

void AIProcessorPlugin::onViewCharts() {
  QProcess::execute("taskkill", QStringList() << "/F" << "/IM" << "tensorboard.exe");
  QString aiTrainingDir = AppConfig::instance().aiTrainingDir();
  QString runsDir = QDir::cleanPath(aiTrainingDir + "/runs");
  runsDir = QDir::toNativeSeparators(runsDir);

  QProcess::startDetached("python",
                          QStringList() << "-m" << "tensorboard.main"
                                        << "--logdir" << runsDir << "--port" << "6006",
                          QDir::cleanPath(aiTrainingDir));

  const int totalSeconds = 12;

  if (!m_progressDialog) return;

  m_progressDialog->setLabelText(
      m_ctx->translate("aiproc.tb_seconds").arg(totalSeconds));
  m_progressDialog->setValue(0);
  m_progressDialog->show();
  m_progressDialog->centerOnWidget(); // Centers on MainWindow by default
  QApplication::processEvents();

  // Shared countdown state via QTimer on heap to avoid dangling refs
  QTimer *timer = new QTimer(m_progressDialog);
  int *remaining = new int(totalSeconds);

  connect(m_progressDialog, &CustomProgressDialog::stopRequested, timer,
          [timer, remaining, this]() {
            timer->stop();
            timer->deleteLater();
            delete remaining;
            if (m_progressDialog) m_progressDialog->hide();
            ModernMessageBox::information(
                m_ctx->mainWindow(), m_ctx->translate("aiproc.tb_cancelled"),
                m_ctx->translate("aiproc.tb_cancelled_desc"));
          });

  timer->setInterval(1000);
  connect(timer, &QTimer::timeout, this,
      [this, timer, remaining, totalSeconds]() {
          (*remaining)--;
          if (m_progressDialog) {
              int progressValue = ((totalSeconds - *remaining) * 100) / totalSeconds;
              m_progressDialog->setValue(progressValue);
          }
          if (*remaining <= 0) {
              timer->stop();
              timer->deleteLater();
              delete remaining;
              m_progressDialog->hide();
              m_progressDialog->reset();
              QDesktopServices::openUrl(QUrl("http://localhost:6006/"));
          } else {
              if (m_progressDialog) {
                  m_progressDialog->setLabelText(
                      m_ctx->translate("aiproc.tb_seconds").arg(*remaining));
              }
          }
      });

  timer->start();
}

void AIProcessorPlugin::onImageChanged(int index, int total) {
    QString mode = m_ctx->viewer()->getCurrentAIMode();
    if (mode == AIMode::Detection) {
        onObjectDetection();
    } else if (mode == AIMode::Segmentation) {
        onSegmentation();
    }
}

void AIProcessorPlugin::onObjectTracking() {
    if (!m_aiSvc || !m_aiSvc->isTrackingReady()) {
        ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.error"), m_ctx->translate("aiproc.model_not_loaded_warn"));
        return;
    }

    QString lastUsedPath = m_ctx->settings()->getLastUsedPath("ai_video_tracking");
    QString videoPath = QFileDialog::getOpenFileName(m_ctx->mainWindow(), m_ctx->translate("aiproc.select_video_tracking"), lastUsedPath, "Video Files (*.mp4 *.avi *.mkv)");
    if (videoPath.isEmpty()) return;
    
    m_ctx->settings()->setLastUsedPath("ai_video_tracking", QFileInfo(videoPath).absolutePath());

    // Clear current state
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->resetToSingleRenderer();
    m_ctx->viewer()->setAIMode(AIMode::Tracking);
    
    // Stop existing tracking if any
    if (m_trackerThread) {
        m_trackerThread->stop();
        m_trackerThread->wait();
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
        if (m_pauseTrackingAct) m_pauseTrackingAct->setText("Resume");
        if (m_ribbonUI) m_ribbonUI->btnPauseTrack()->setText("Resume");
    } else {
        m_trackerThread->resume();
        if (m_pauseTrackingAct) m_pauseTrackingAct->setText("Pause");
        if (m_ribbonUI) m_ribbonUI->btnPauseTrack()->setText("Pause");
    }
}

void AIProcessorPlugin::onStopTracking() {
    if (m_trackerThread) {
        m_trackerThread->stop();
        m_trackerThread->wait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }
    m_ctx->viewer()->setAIMode(AIMode::None);
    updateActions();
}

void AIProcessorPlugin::onTrackingFrameReceived(const cv::Mat& frame) {
    if (m_ctx->viewer()->getCurrentAIMode() != AIMode::Tracking) return;
    m_ctx->scene()->setTextureActor(Image2DLoader::loadFromMat(frame));
    m_ctx->scene()->vtkWidget()->renderWindow()->Render();
}

void AIProcessorPlugin::onTrackingFinished() {
    if (m_trackerThread) {
        m_trackerThread->wait();
        delete m_trackerThread;
        m_trackerThread = nullptr;
    }
    m_ctx->viewer()->setAIMode(AIMode::None);
    ModernMessageBox::information(m_ctx->mainWindow(), "Tracking Finished", "Video tracking completed.");
    updateActions();
}

void AIProcessorPlugin::updateActions() {
    QMenu *aiMenu = m_ctx->getMenu("ai_menu");
    if (aiMenu) {
        aiMenu->setTitle(m_ctx->translate("ai.menu"));
    }
    if (!m_ctx->mainWindow() || !m_runDetAct) return;
  QString mode = m_ctx->viewer()->getCurrentAIMode();
  bool has2D = !m_ctx->viewer()->getCurrent2DImagePath().isEmpty();

  bool detReady = m_aiSvc && m_aiSvc->isDetectionReady();
  bool segReady = m_aiSvc && m_aiSvc->isSegmentationReady();
  bool trackReady = m_aiSvc && m_aiSvc->isTrackingReady();
  bool isTrackingMode = (mode == AIMode::Tracking);

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

  if (m_ribbonUI) {
      m_ribbonUI->btnHide()->setText(m_ctx->translate("ai.hide_results"));
      m_ribbonUI->btnHide()->setEnabled(mode != AIMode::None);
      m_ribbonUI->btnTrack()->setText(m_ctx->translate("ai.run_tracking"));
      m_ribbonUI->btnTrain()->setText(m_ctx->translate("ai.training"));
      m_ribbonUI->btnChart()->setText(m_ctx->translate("ai.charts"));
      
      // Update GroupBox titles
      if (QLabel *lbl = m_ribbonUI->groupAI()->findChild<QLabel*>("groupTitleLabel")) {
          lbl->setText(m_ctx->translate("ai.menu"));
      }
      if (QLabel *lbl = m_ribbonUI->groupTrain()->findChild<QLabel*>("groupTitleLabel")) {
          lbl->setText(m_ctx->translate("ai.training"));
      }
  }
}
