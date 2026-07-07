#include "ViewerPlugin.h"
#include "MainWindow.h"
#include "IAppContext.h"
#include "ISettingsService.h"
#include "IViewerService.h"
#include "ISceneService.h"
#include "SignalBus.h"
#include "Image2DLoader.h"
#include "Model3DLoader.h"
#include "DicomLoader.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QMenu>
#include <QDir>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QStyle>
#include <QPainter>
#include <QLinearGradient>
#include "LanguageManager.h"
#include "../../utils/CustomProgressDialog.h"
#include <QApplication>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <QListWidget>

#include "ViewerRibbonUI.h"
#include "ViewerNavigatorUI.h"
#include "ViewerListUI.h"
#include "ViewerViewModel.h"

void ViewerPlugin::initialize(IAppContext* context) {
    m_ctx = context;
    m_progressDialog = new CustomProgressDialog(m_ctx->mainWindow());
    m_viewModel = new ViewerViewModel(m_ctx, this);
    
    connect(m_viewModel, &ViewerViewModel::loadingStarted, this, [this](const QString& msg){
        if (m_progressDialog) {
            m_progressDialog->setLabelText(msg);
            m_progressDialog->setRange(0, 100);
            m_progressDialog->setValue(0);
            m_progressDialog->show();
            m_progressDialog->centerOnWidget();
            QApplication::processEvents();
        }
    });
    connect(m_viewModel, &ViewerViewModel::progressUpdated, this, [this](int percent){
        if (m_progressDialog) {
            m_progressDialog->setValue(percent);
            QApplication::processEvents();
        }
    });
    connect(m_viewModel, &ViewerViewModel::loadingFinished, this, [this](){
        if (m_progressDialog) m_progressDialog->hide();
    });
    connect(m_viewModel, &ViewerViewModel::showNavigationUI, this, [this](bool show){
        if (m_navUI && m_navUI->widget()) {
            if (show) m_navUI->widget()->show();
            else m_navUI->widget()->hide();
        }
    });
    
    QMenu* viewerMenu = m_ctx->getMenu("viewer_menu");
    viewerMenu->clear();
    
    m_load2DAct = viewerMenu->addAction(m_ctx->translate("viewer.load_2d"), this, &ViewerPlugin::onLoad2DImages);
    m_load3DAct = viewerMenu->addAction(m_ctx->translate("viewer.load_3d"), this, &ViewerPlugin::onLoad3DImages);
    m_loadDicomAct = viewerMenu->addAction(m_ctx->translate("viewer.load_dicom"), this, &ViewerPlugin::onLoadDicom);

    m_navUI = new ViewerNavigatorUI(m_ctx, this);
    m_listUI = new ViewerListUI(m_ctx, this);
    m_listUI->setupUI();

    connect(m_navUI->btnPrev(), &QPushButton::clicked, this, &ViewerPlugin::onPrevImage); 
    connect(m_navUI->btnNext(), &QPushButton::clicked, this, &ViewerPlugin::onNextImage);
    connect(m_navUI->btnAutoPrev(), &QPushButton::clicked, this, &ViewerPlugin::onAutoPrev); 
    connect(m_navUI->btnAutoNext(), &QPushButton::clicked, this, &ViewerPlugin::onAutoNext);

    // Inject Load/Data tools into tab.view panel
    if (QWidget* panel = m_ctx->getTabPanel("tab.view")) {
        m_ribbonUI = new ViewerRibbonUI(m_ctx, panel, this);
        connect(m_ribbonUI->btnLoad2D(), &QToolButton::clicked, this, &ViewerPlugin::onLoad2DImages);
        connect(m_ribbonUI->btnLoad3D(), &QToolButton::clicked, this, &ViewerPlugin::onLoad3DImages);
        connect(m_ribbonUI->btnLoadDicom(), &QToolButton::clicked, this, &ViewerPlugin::onLoadDicom);
    }

    connect(m_ctx->signalBus(), &SignalBus::imageIndexChanged, this, &ViewerPlugin::onImageIndexChanged);
    connect(m_ctx->signalBus(), &SignalBus::autoNavigationChanged, this, &ViewerPlugin::onAutoNavigationChanged);
    connect(m_ctx->signalBus(), &SignalBus::stateChanged, this, &ViewerPlugin::updateActions);
    connect(m_ctx->signalBus(), &SignalBus::languageChanged, this, &ViewerPlugin::updateActions);
    
    updateActions();
}

void ViewerPlugin::cleanup() {
}

void ViewerPlugin::onLoad2DImages() {
  QString lastUsedPath = m_ctx->settings()->getLastUsedPath("viewer_2d");
  QString fileName = QFileDialog::getOpenFileName(m_ctx->mainWindow(), m_ctx->translate("file.select_2d"), lastUsedPath, "Images (*.png *.jpg *.jpeg *.bmp)"); 
  if (!fileName.isEmpty()) m_viewModel->load2DImage(fileName);
}

void ViewerPlugin::onLoad3DImages() {
  QString lastUsedPath = m_ctx->settings()->getLastUsedPath("viewer_3d");
  QString obj = QFileDialog::getOpenFileName(m_ctx->mainWindow(), m_ctx->translate("file.select_3d"), lastUsedPath, "OBJ Files (*.obj)"); 
  if (!obj.isEmpty()) m_viewModel->load3DModel(obj);
}

void ViewerPlugin::onLoadDicom() {
  QString lastUsedPath = m_ctx->settings()->getLastUsedPath("viewer_dicom");
  QString dir = QFileDialog::getExistingDirectory(m_ctx->mainWindow(), m_ctx->translate("file.select_dicom"), lastUsedPath, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty()) m_viewModel->loadDicom(dir);
}

void ViewerPlugin::onPrevImage() {
    m_ctx->viewer()->onPrevImage();
}

void ViewerPlugin::onNextImage() {
    m_ctx->viewer()->onNextImage();
}

void ViewerPlugin::onAutoPrev() {
    m_ctx->viewer()->onAutoPrev();
}

void ViewerPlugin::onAutoNext() {
    m_ctx->viewer()->onAutoNext();
}

void ViewerPlugin::onImageIndexChanged(int index, int total) {
    bool hasImages = (total > 0);
    if (m_navUI) {
        m_navUI->btnPrev()->setEnabled(index > 0);
        m_navUI->btnNext()->setEnabled(index >= 0 && index < total - 1);
        m_navUI->btnAutoPrev()->setEnabled(hasImages);
        m_navUI->btnAutoNext()->setEnabled(hasImages);
    }
}

void ViewerPlugin::onAutoNavigationChanged(bool active, bool isNext) {
    if (m_navUI) {
        m_navUI->btnAutoNext()->setChecked(active && isNext);
        m_navUI->btnAutoPrev()->setChecked(active && !isNext);
    }
}

void ViewerPlugin::updateActions() {
    QMenu* viewerMenu = m_ctx->getMenu("viewer_menu");
    if (viewerMenu) {
        viewerMenu->setTitle(m_ctx->translate("viewer.menu"));
    }
    if (m_load2DAct) {
        m_load2DAct->setText(m_ctx->translate("viewer.load_2d"));
    }
    if (m_load3DAct) {
        m_load3DAct->setText(m_ctx->translate("viewer.load_3d"));
    }
    if (m_loadDicomAct) {
        m_loadDicomAct->setText(m_ctx->translate("viewer.load_dicom"));
    }
    if (m_navUI) {
        m_navUI->btnPrev()->setText(m_ctx->translate("viewer.prev"));
        m_navUI->btnAutoPrev()->setText(m_ctx->translate("viewer.auto_prev"));
        m_navUI->btnAutoNext()->setText(m_ctx->translate("viewer.auto_next"));
        m_navUI->btnNext()->setText(m_ctx->translate("viewer.next"));
    }
    if (m_ribbonUI) {
        m_ribbonUI->btnLoad2D()->setText(m_ctx->translate("viewer.load_2d"));
        m_ribbonUI->btnLoad3D()->setText(m_ctx->translate("viewer.load_3d"));
        m_ribbonUI->btnLoadDicom()->setText(m_ctx->translate("viewer.load_dicom"));
        if (QLabel *lbl = m_ribbonUI->groupView()->findChild<QLabel*>("groupTitleLabel")) {
            lbl->setText(m_ctx->translate("viewer.menu"));
        }
    }
}
