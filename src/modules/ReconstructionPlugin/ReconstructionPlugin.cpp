#include "ReconstructionPlugin.h"
#include "IAppContext.h"
#include "ISettingsService.h"
#include "IViewerService.h"
#include "ISceneService.h"
#include "IReconstructionService.h"
#include "Image2DLoader.h"
#include "SignalBus.h"
#include "ReconstructThread.h"
#include "LanguageManager.h"
#include "../../utils/CustomProgressDialog.h"
#include "../../utils/ModernMessageBox.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QApplication>
#include <QDir>
#include <QMenu>
#include <QPointer>
#include <QPainter>
#include <QLinearGradient>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkPointData.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <QVTKOpenGLNativeWidget.h>
#include <QMainWindow>

#include <QGroupBox>
#include <QToolButton>
#include <QListWidget>
#include <QListWidgetItem>
#include <QStyle>
#include "ReconstructionRibbonUI.h"

void ReconstructionPlugin::initialize(IAppContext* context) {
    m_ctx = context;

    // Look up IReconstructionService từ ServiceRegistry — không dùng concrete type
    m_reconSvc = m_ctx->services()->get<IReconstructionService>();
    if (!m_reconSvc) {
        qWarning() << "[ReconstructionPlugin] IReconstructionService not found in ServiceRegistry!";
    }

    m_progressDialog = new CustomProgressDialog(m_ctx->mainWindow());
    connect(m_progressDialog, &CustomProgressDialog::stopRequested,
            this, &ReconstructionPlugin::onProgressStopped);

    QMenu* reconMenu = m_ctx->getMenu("recon_menu");
    reconMenu->clear();

    QAction *loadAct = reconMenu->addAction(m_ctx->translate("recon.load_images"), this, &ReconstructionPlugin::onLoadMultipleImages);
    QAction *runAct  = reconMenu->addAction(m_ctx->translate("recon.start"), this, &ReconstructionPlugin::onRunReconstruction);
    reconMenu->addSeparator();
    m_toggleCloudAct = reconMenu->addAction(m_ctx->translate("recon.view_model"), this, &ReconstructionPlugin::onTogglePointCloud);

    m_ribbonUI = new ReconstructionRibbonUI(m_ctx, this);
    m_ribbonUI->setupUI();
    
    if (m_ribbonUI->btnLoad()) connect(m_ribbonUI->btnLoad(), &QToolButton::clicked, this, &ReconstructionPlugin::onLoadMultipleImages);
    if (m_ribbonUI->btnRun()) connect(m_ribbonUI->btnRun(), &QToolButton::clicked, this, &ReconstructionPlugin::onRunReconstruction);
    if (m_ribbonUI->btnToggleCloud()) connect(m_ribbonUI->btnToggleCloud(), &QToolButton::clicked, this, &ReconstructionPlugin::onTogglePointCloud);

    connect(m_ctx->signalBus(), &SignalBus::stateChanged, this, &ReconstructionPlugin::updateActions);
    connect(m_ctx->signalBus(), &SignalBus::languageChanged, this, &ReconstructionPlugin::updateActions);

    updateActions();
}

void ReconstructionPlugin::cleanup() {
    if (m_currentReconstructThread) {
        m_currentReconstructThread->terminate();
        m_currentReconstructThread->wait();
    }
}
void ReconstructionPlugin::onLoadMultipleImages() {
    QString lastUsedPath = m_ctx->settings()->getLastUsedPath("recon");
    QString folderPath = QFileDialog::getExistingDirectory(m_ctx->mainWindow(), m_ctx->translate("file.select_recon"), lastUsedPath);
    
    if (folderPath.isEmpty()) {
        return;
    }
    
    QDir dir(folderPath);
    QStringList fileNames = dir.entryList({"*.png", "*.jpg", "*.jpeg", "*.bmp"}, QDir::Files | QDir::NoSymLinks, QDir::Name);
    for (int i = 0; i < fileNames.size(); ++i) {
        fileNames[i] = dir.absoluteFilePath(fileNames[i]);
    }

    if (fileNames.isEmpty()) {
        ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.error"), m_ctx->translate("recon.load_min_images_error"));
        return;
    }

    if (m_progressDialog) {
        m_progressDialog->setLabelText(m_ctx->translate("recon.loading_images"));
        m_progressDialog->setRange(0, 0);
        m_progressDialog->show();
        m_progressDialog->centerOnWidget();
        QApplication::processEvents();
    }

    m_ctx->settings()->setLastUsedPath("recon", QFileInfo(fileNames.first()).absolutePath());
    QStringList stdFiles;
    for (const QString& f : fileNames) {
        stdFiles.push_back(QDir::toNativeSeparators(f));
    }
    std::vector<QString> paths;
    for (const auto &f : stdFiles) paths.push_back(f);

    // Dùng IReconstructionService interface — không biết ReconstructionPipeline
    if (m_reconSvc) m_reconSvc->setImages(stdFiles);
    
    // Disable updating 2D texture actor during navigation for 3D Reconstruction mode
    m_ctx->viewer()->setUpdateSceneEnabled(false);
    
    // Reset AI mode to clear any active video tracking
    m_ctx->viewer()->setAIMode(AIMode::None);
    
    // Push file list to viewer service. ViewerPlugin will automatically show the thumbnail list.
    QStringList filenamesOnly;
    for (const QString& f : fileNames) {
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
    vtkSmartPointer<vtkActor> actor = Image2DLoader::load(fileNames.first());
    if (actor) {
        m_ctx->scene()->setTextureActor(actor);
        m_ctx->scene()->renderer()->ResetCamera();
        m_ctx->scene()->vtkWidget()->renderWindow()->Render();
    }

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
    if (paramsPath.isEmpty()) {
        paramsPath = QFileDialog::getOpenFileName(m_ctx->mainWindow(), m_ctx->translate("file.select_camera"), folder, "Text Files (*.txt)");
    }

    if (!paramsPath.isEmpty() && m_reconSvc) {
        m_reconSvc->loadCameraParams(paramsPath);
    }

    if (m_progressDialog) {
        m_progressDialog->hide();
    }

    if(fileNames.size() > 2 && !paramsPath.isEmpty()){
        ModernMessageBox::information(m_ctx->mainWindow(), m_ctx->translate("recon.load_success_title"), m_ctx->translate("recon.load_success_msg").arg(fileNames.size()).arg(paramsPath));
    }
    else if(fileNames.size() < 2 && !paramsPath.isEmpty())
    {
        ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.error"), m_ctx->translate("recon.load_min_images_error"));
    }
    else if(paramsPath.isEmpty()){
        ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("common.error"), m_ctx->translate("recon.load_camera_params_error"));
    }
}

void ReconstructionPlugin::onRunReconstruction() {
    if (!m_reconSvc || m_reconSvc->getImageList().isEmpty()) {
        onLoadMultipleImages();
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
    QApplication::processEvents();

    // Lấy IReconstructionService interface thay vì pipeline cụ thể
    if (!m_reconSvc) return;

    m_currentReconstructThread = new ReconstructThread(m_reconSvc, this);
    QPointer<CustomProgressDialog> progressPtr(m_progressDialog);
    connect(m_currentReconstructThread, &QThread::finished, this, [this, progressPtr]() {
        if (progressPtr) progressPtr->hide();
        if (!m_currentReconstructThread) return;

        bool success  = m_currentReconstructThread->isSuccess();
        int pointCount = m_reconSvc ? (int)m_reconSvc->getPointCloud().size() : 0;
        m_currentReconstructThread->deleteLater();
        m_currentReconstructThread = nullptr;

        // Emit lifecycle signal via SignalBus
        emit m_ctx->signalBus()->reconstructionFinished(success, pointCount);

        if (!success || pointCount == 0) {
            ModernMessageBox::warning(m_ctx->mainWindow(),
                m_ctx->translate("recon.failed_title"),
                m_ctx->translate("recon.failed_msg"));
        } else {
            onTogglePointCloud(true);
            ModernMessageBox::information(m_ctx->mainWindow(),
                m_ctx->translate("recon.success_title"),
                m_ctx->translate("recon.success_msg").arg(pointCount));
        }
    }, Qt::QueuedConnection);

    emit m_ctx->signalBus()->reconstructionStarted();
    m_currentReconstructThread->start();
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
    auto pts    = m_reconSvc->getPointCloud();
    auto colors = m_reconSvc->getPointColors();
    if (pts.empty()) return;

    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->resetToSingleRenderer();

    vtkNew<vtkPoints> vP;
    vtkNew<vtkCellArray> vV;
    vtkNew<vtkUnsignedCharArray> vC;
    vC->SetNumberOfComponents(3);

    for (size_t i = 0; i < pts.size(); ++i) {
        vP->InsertNextPoint(pts[i].x, pts[i].y, pts[i].z);
        vV->InsertNextCell(1);
        vV->InsertCellPoint(i);
        if (i < colors.size()) vC->InsertNextTuple3(colors[i][2], colors[i][1], colors[i][0]);
        else vC->InsertNextTuple3(255, 255, 255);
    }

    vtkNew<vtkPolyData> pd;
    pd->SetPoints(vP);
    pd->SetVerts(vV);
    pd->GetPointData()->SetScalars(vC);

    vtkNew<vtkVertexGlyphFilter> gf;
    gf->SetInputData(pd);
    vtkNew<vtkPolyDataMapper> m;
    m->SetInputConnection(gf->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(m);
    actor->GetProperty()->SetPointSize(3);

    m_ctx->scene()->setPointCloudActor(actor);
    m_ctx->scene()->renderer()->ResetCamera();
    m_ctx->scene()->vtkWidget()->renderWindow()->Render();
    m_ctx->updateMenuStates();
}

void ReconstructionPlugin::onHidePointCloud() {
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->setPointCloudVisible(false);
    m_ctx->updateMenuStates();
    m_ctx->scene()->vtkWidget()->renderWindow()->Render();
}

void ReconstructionPlugin::onProgressStopped() {
    if (m_progressDialog) m_progressDialog->hide();

    if (m_currentReconstructThread) {
        m_currentReconstructThread->terminate();
        m_currentReconstructThread->wait();
        m_currentReconstructThread->deleteLater();
        m_currentReconstructThread = nullptr;
        ModernMessageBox::information(m_ctx->mainWindow(), m_ctx->translate("recon.cancelled"), m_ctx->translate("recon.cancelled_desc"));
    }
}

void ReconstructionPlugin::updateActions() {
    QMenu* reconMenu = m_ctx->getMenu("recon_menu");
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
