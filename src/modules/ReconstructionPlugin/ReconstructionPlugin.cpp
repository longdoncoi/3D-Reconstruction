#include "ReconstructionPlugin.h"
#include "IAppContext.h"
#include "IShellService.h"
#include "AppConstants.h"
#include "ISettingsService.h"
#include "IViewerService.h"
#include "ISceneService.h"
#include "IReconstructionService.h"
#include "SignalBus.h"
// Architecture note: the progress dialog is consumed through IProgressDialog
// via IShellService; the concrete AppCore widget stays out of the plugin ABI.
#include "IProgressDialog.h"
#include "../../utils/ModernMessageBox.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include <QMenu>
#include <QPointer>
#include <QPainter>
#include <QLinearGradient>
#include <utility>
#include <QMainWindow>

#include <QGroupBox>
#include <QToolButton>
#include <QListWidget>
#include <QListWidgetItem>
#include <QStyle>
#include "ReconstructionRibbonUI.h"
#include "IAppPaths.h"
#include "IUserManager.h"

bool ReconstructionPlugin::initialize(IAppContext* context) {
    m_ctx = context;

    // Look up IReconstructionService từ ServiceRegistry — không dùng concrete type
    m_reconSvc = m_ctx->services()->getRequired<IReconstructionService>();
    if (!m_reconSvc) {
        qWarning() << "[ReconstructionPlugin] IReconstructionService not found in ServiceRegistry!";
    }

    m_progressDialog = m_ctx->shell()->getProgressDialog();
    connect(m_progressDialog, &IProgressDialog::stopRequested,
            this, &ReconstructionPlugin::onProgressStopped);

    // Reconstruction now runs inside the service's own worker; the plugin
    // only reacts to the interface signal (connected once, app lifetime).
    connect(m_reconSvc, &IReconstructionService::reconstructFinished, this,
            [this](bool success, const QString& errorMessage) {
        if (m_progressDialog) m_progressDialog->hide();
        if (m_cancelRequested) {
            m_cancelRequested = false;
            return; // "cancelled" notice already shown by onProgressStopped
        }

        const int pointCount = m_reconSvc ? static_cast<int>(m_reconSvc->pointCount()) : 0;

        // Emit lifecycle signal via SignalBus
        emit m_ctx->signalBus()->reconstructionFinished(success, pointCount);

        if (!success || pointCount == 0) {
            QString msg = m_ctx->translate("recon.failed_msg");
            if (!errorMessage.isEmpty())
                msg += QStringLiteral("\n\n") + errorMessage; // chẩn đoán từ pipeline
            ModernMessageBox::warning(m_ctx->shell()->dialogParent(),
                m_ctx->translate("recon.failed_title"), msg);
        } else {
            onTogglePointCloud(true);
            ModernMessageBox::information(m_ctx->shell()->dialogParent(),
                m_ctx->translate("recon.success_title"),
                m_ctx->translate("recon.success_msg").arg(pointCount));
        }
    }, Qt::QueuedConnection);

    QMenu* reconMenu = m_ctx->shell()->getMenu(AppConstants::MenuIds::Reconstruction);
    reconMenu->clear();

    reconMenu->addAction(m_ctx->translate("recon.load_images"), this, &ReconstructionPlugin::onLoadMultipleImages);
    reconMenu->addAction(m_ctx->translate("recon.start"), this, &ReconstructionPlugin::onRunReconstruction);
    reconMenu->addSeparator();
    m_toggleCloudAct = reconMenu->addAction(m_ctx->translate("recon.view_model"), this, &ReconstructionPlugin::onTogglePointCloud);

    m_ribbonUI = new ReconstructionRibbonUI(m_ctx, this);
    m_ribbonUI->setupUI();
    
    if (m_ribbonUI->btnLoad()) connect(m_ribbonUI->btnLoad(), &QToolButton::clicked, this, &ReconstructionPlugin::onLoadMultipleImages);
    if (m_ribbonUI->btnRun()) connect(m_ribbonUI->btnRun(), &QToolButton::clicked, this, &ReconstructionPlugin::onRunReconstruction);
    if (m_ribbonUI->btnToggleCloud()) connect(m_ribbonUI->btnToggleCloud(), &QToolButton::clicked, this, &ReconstructionPlugin::onTogglePointCloud);

    connect(m_ctx->signalBus(), &SignalBus::stateChanged, this, &ReconstructionPlugin::updateActions);
    connect(m_ctx->signalBus(), &SignalBus::languageChanged, this, &ReconstructionPlugin::updateActions);
    connect(m_ctx->signalBus(), &SignalBus::agentUiActionRequested, this,
            [this](const QString &action, const QVariantMap &parameters) {
        // "folder" is filled by AgentActionManifest::applySampleDefaults()
        // from Config/agent_action_manifest.json when the agent provides none.
        const QString folder = parameters.value("folder").toString();
        bool handled = false;
        if (action == "reconstruction.load_images") {
            if (!folder.isEmpty()) {
                m_agentImageFolder = folder;
                onLoadMultipleImages();
                handled = true;
            }
        } else if (action == "reconstruction.start_reconstruction") {
            // A workflow start follows reconstruction.load_images.  Suppress
            // the interactive/default load fallback so the success dialog is
            // not shown again for the same input set.
            m_skipImageLoadForAgentRun = parameters.contains("workflow_id");
            onRunReconstruction();
            handled = m_reconSvc && !m_reconSvc->getImageList().isEmpty();
        } else if (action == "reconstruction.view_3d_model") {
            onTogglePointCloud(true);
            handled = true;
        } else if (action == "reconstruction.close_3d_model") {
            onHidePointCloud();
            handled = true;
        }
        const QString requestId = parameters.value("request_id").toString();
        if (!requestId.isEmpty() && action.startsWith("reconstruction."))
            emit m_ctx->signalBus()->agentUiActionCompleted(requestId, handled,
                QVariantMap{{"action", action}, {"error", handled ? "" : "Action was not handled"}});
    });

    updateActions();
    return true;
}

void ReconstructionPlugin::cleanup() {
    // Service owns the worker thread now; just make it stop.
    if (m_reconSvc) m_reconSvc->stopReconstruction();
}
void ReconstructionPlugin::onLoadMultipleImages() {
    QString folderPath = m_agentImageFolder;
    const bool useAgentFolder = !folderPath.isEmpty();
    m_agentImageFolder.clear();
    if (!useAgentFolder) {
        const QString lastUsedPath = m_ctx->settings()->getLastUsedPath("recon");
        folderPath = QFileDialog::getExistingDirectory(m_ctx->shell()->dialogParent(), m_ctx->translate("file.select_recon"), lastUsedPath);
    }
    
    if (folderPath.isEmpty()) {
        return;
    }
    
    QDir dir(folderPath);
    QStringList fileNames = dir.entryList({"*.png", "*.jpg", "*.jpeg", "*.bmp"}, QDir::Files | QDir::NoSymLinks, QDir::Name);
    for (int i = 0; i < fileNames.size(); ++i) {
        fileNames[i] = dir.absoluteFilePath(fileNames[i]);
    }

    if (fileNames.isEmpty()) {
        ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"), m_ctx->translate("recon.load_min_images_error"));
        return;
    }

    if (m_progressDialog) {
        m_progressDialog->setLabelText(m_ctx->translate("recon.loading_images"));
        m_progressDialog->setRange(0, 0);
        m_progressDialog->show();
        m_progressDialog->centerOnWidget();
        // Repaint without re-entering user input (see ViewerPlugin).
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    m_ctx->settings()->setLastUsedPath("recon", QFileInfo(fileNames.first()).absolutePath());
    QStringList stdFiles;
    for (const QString& f : std::as_const(fileNames)) {
        stdFiles.push_back(QDir::toNativeSeparators(f));
    }
    std::vector<QString> paths;
    for (const auto &f : std::as_const(stdFiles)) paths.push_back(f);

    // Dùng IReconstructionService interface — không biết ReconstructionPipeline
    if (m_reconSvc) m_reconSvc->setImages(stdFiles);
    
    // Disable updating 2D texture actor during navigation for 3D Reconstruction mode
    m_ctx->viewer()->setUpdateSceneEnabled(false);
    
    // Reset AI mode to clear any active video tracking
    m_ctx->viewer()->setAIMode(AIMode::None);
    
    // Push file list to viewer service. ViewerPlugin will automatically show the thumbnail list.
    QStringList filenamesOnly;
    for (const QString& f : std::as_const(fileNames)) {
        filenamesOnly.append(QFileInfo(f).fileName());
    }
    m_ctx->viewer()->setCurrent2DImagePath(fileNames.first());
    m_ctx->viewer()->setImageList(filenamesOnly, 0);

    // Immediately clean up any loaded DICOM volume/renderers to free up RAM/VRAM
    m_ctx->scene()->resetToSingleRenderer();
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->clear2DTexture();

    // Display the first image on the VTK scene
    m_ctx->scene()->showImage(fileNames.first());

    // Try to load camera params automatically
    QFileInfo firstFile(fileNames.first());
    QString folder = firstFile.absolutePath();
    QString paramsPath;

    // 1. Check folder name + _par.txt (e.g. folder "temple" -> "temple_par.txt")
    QDir paramDir(folder);
    QStringList filters;
    filters << "*_par.txt";
    QStringList parFiles = paramDir.entryList(filters, QDir::Files);
    if (!parFiles.isEmpty()) {
        paramsPath = folder + "/" + parFiles.first();
    } else {
        paramsPath.clear();
    }

    // 2. If not found, ask user
    if (paramsPath.isEmpty() && !useAgentFolder) {
        paramsPath = QFileDialog::getOpenFileName(m_ctx->shell()->dialogParent(), m_ctx->translate("file.select_camera"), folder, "Text Files (*.txt)");
    }

    if (!paramsPath.isEmpty() && m_reconSvc) {
        m_reconSvc->loadCameraParams(paramsPath);
    }

    if (m_progressDialog) {
        m_progressDialog->hide();
    }

    if(fileNames.size() > 2 && !paramsPath.isEmpty()){
        ModernMessageBox::information(m_ctx->shell()->dialogParent(), m_ctx->translate("recon.load_success_title"), m_ctx->translate("recon.load_success_msg").arg(fileNames.size()).arg(paramsPath));
    }
    else if(fileNames.size() < 2 && !paramsPath.isEmpty())
    {
        ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"), m_ctx->translate("recon.load_min_images_error"));
    }
    else if(paramsPath.isEmpty()){
        ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("common.error"), m_ctx->translate("recon.load_camera_params_error"));
    }
}

void ReconstructionPlugin::onRunReconstruction() {
    const bool skipImageLoad = m_skipImageLoadForAgentRun;
    m_skipImageLoadForAgentRun = false;
    if (!m_reconSvc || m_reconSvc->getImageList().isEmpty()) {
        if (!skipImageLoad) onLoadMultipleImages();
    }
    if (!m_reconSvc || m_reconSvc->getImageList().isEmpty()) return;

    // Stop any active video tracking before reconstruction
    m_ctx->viewer()->setAIMode(AIMode::None);

    // Immediately clean up any loaded DICOM volume/renderers to free up RAM/VRAM
    m_ctx->scene()->resetToSingleRenderer();
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->clear2DTexture();

    if (!m_progressDialog) return;
    
    m_progressDialog->setLabelText(m_ctx->translate("recon.reconstructing"));
    m_progressDialog->setRange(0, 0);
    m_progressDialog->show();
    m_progressDialog->centerOnWidget(); // Centers on MainWindow by default
    // Repaint without re-entering user input (see ViewerPlugin).
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);

    // Lấy IReconstructionService interface thay vì pipeline cụ thể
    if (!m_reconSvc) return;

    // Apply configuration from Settings
    auto* um = m_ctx->services()->get<IUserManager>();
    if (um) {
        QString username = um->currentUsername();
        int density = um->getUserPref(username, "recon_density", "1").toInt();
        bool autoClean = um->getUserPref(username, "recon_auto_clean", "true") == "true";
        int maxFeatures = um->getUserPref(username, "recon_max_features", "10000").toInt();
        m_reconSvc->configure(density, maxFeatures, autoClean);
    }

    m_cancelRequested = false;
    emit m_ctx->signalBus()->reconstructionStarted();
    if (!m_reconSvc->reconstruct()) {
        // Busy: another reconstruction started and was never stopped.
        if (m_progressDialog) m_progressDialog->hide();
        ModernMessageBox::warning(m_ctx->shell()->dialogParent(),
            m_ctx->translate("recon.failed_title"),
            m_ctx->translate("recon.busy_msg"));
    }
}

void ReconstructionPlugin::onTogglePointCloud(bool forceShow) {
    if (m_ctx->scene()->isPointCloudVisible() && !forceShow) {
        onHidePointCloud();
        return;
    }
    onShowPointCloud();
}

void ReconstructionPlugin::onShowPointCloud() {
    if (!m_reconSvc) return;
    const auto snapshot = m_reconSvc->pointCloudSnapshot();
    if (!snapshot || snapshot->points.empty()) {
        QMessageBox::warning(m_ctx->shell()->dialogParent(), 
                             m_ctx->translate("recon.failed_title"), 
                             m_ctx->translate("recon.no_model_error"));
        return;
    }

    // The snapshot keeps points and colours from the same run alive for the
    // duration of the call, even if a new reconstruction starts concurrently.
    m_ctx->scene()->setPointCloudData(snapshot->points, snapshot->colors);
    m_ctx->shell()->updateMenuStates();
}

void ReconstructionPlugin::onHidePointCloud() {
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->setPointCloudVisible(false);
    m_ctx->shell()->updateMenuStates();
    m_ctx->scene()->render();
}

void ReconstructionPlugin::onProgressStopped() {
    if (m_progressDialog) m_progressDialog->hide();

    if (m_reconSvc && m_reconSvc->isRunning()) {
        m_cancelRequested = true;
        m_reconSvc->stopReconstruction();
        ModernMessageBox::information(m_ctx->shell()->dialogParent(),
            m_ctx->translate("recon.cancelled"), m_ctx->translate("recon.cancelled_desc"));
    }
}

void ReconstructionPlugin::updateActions() {
    QMenu* reconMenu = m_ctx->shell()->getMenu(AppConstants::MenuIds::Reconstruction);
    if (reconMenu) {
        reconMenu->setTitle(m_ctx->translate("recon.menu"));
    }
    if (m_toggleCloudAct) {
        bool visible = m_ctx->scene()->isPointCloudVisible();
        m_toggleCloudAct->setText(visible ? m_ctx->translate("common.close") + " " + m_ctx->translate("recon.view_model") : m_ctx->translate("recon.view_model"));
        m_toggleCloudAct->setChecked(visible);
    }
    if (m_ribbonUI) {
        m_ribbonUI->updateTranslations();
        m_ribbonUI->updateStates(m_ctx->scene()->isPointCloudVisible());
    }
}
