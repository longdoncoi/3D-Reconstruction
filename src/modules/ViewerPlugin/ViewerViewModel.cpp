#include "ViewerViewModel.h"
#include "Model3DLoader.h"
#include "ISceneService.h"
#include "IViewerService.h"
#include "ISettingsService.h"
#include "Image2DLoader.h"
#include "DicomLoader.h"
#include <QFileInfo>
#include <QDir>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>

ViewerViewModel::ViewerViewModel(IAppContext* ctx, QObject* parent)
    : QObject(parent), m_ctx(ctx) {}

void ViewerViewModel::load2DImage(const QString& filePath) {
    if (filePath.isEmpty()) return;

    emit loadingStarted(m_ctx->translate("viewer.loading_2d"));
    emit progressUpdated(10);

    m_ctx->settings()->setLastUsedPath("viewer_2d", QFileInfo(filePath).absolutePath());
    m_ctx->viewer()->setCurrent2DImagePath(filePath);
    m_ctx->scene()->resetToSingleRenderer();
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->clear2DTexture();
    m_ctx->viewer()->setAIMode(AIMode::None);

    emit progressUpdated(30);

    auto actor = Image2DLoader::load(filePath);
    if (actor) {
        emit progressUpdated(70);
        m_ctx->scene()->setTextureActor(actor);
        m_ctx->scene()->renderer()->ResetCamera();
        m_ctx->scene()->vtkWidget()->renderWindow()->Render();

        QFileInfo fi(filePath);
        QDir dir = fi.dir();
        QStringList imageFileList = dir.entryList(QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp", QDir::Files, QDir::Name);
        int currentIndex = imageFileList.indexOf(fi.fileName());
        m_ctx->viewer()->setImageList(imageFileList, currentIndex);
    }

    emit showNavigationUI(true);
    emit loadingFinished();
    m_ctx->updateMenuStates();
}

void ViewerViewModel::load3DModel(const QString& filePath) {
    if (filePath.isEmpty()) return;

    emit loadingStarted(m_ctx->translate("viewer.loading_3d"));
    emit progressUpdated(20);
    emit showNavigationUI(false);

    QFileInfo fi(filePath);
    m_ctx->settings()->setLastUsedPath("viewer_3d", fi.absolutePath());
    m_ctx->viewer()->setCurrent2DImagePath("");
    m_ctx->viewer()->setImageList(QStringList(), -1);

    emit progressUpdated(50);
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->clear2DTexture();
    m_ctx->scene()->resetToSingleRenderer();

    emit progressUpdated(75);
    // Use the scene's OBJ+MTL loader which owns the VTK pipeline
    QString mtlPath = fi.path() + "/" + fi.completeBaseName() + ".mtl";
    m_ctx->scene()->loadOBJwithMTL(filePath, mtlPath);

    emit loadingFinished();
    m_ctx->updateMenuStates();
}

void ViewerViewModel::loadDicom(const QString& directoryPath) {
    if (directoryPath.isEmpty()) return;

    emit loadingStarted(m_ctx->translate("viewer.loading_dicom"));
    emit progressUpdated(20);
    emit showNavigationUI(false);

    m_ctx->settings()->setLastUsedPath("viewer_dicom", directoryPath);
    m_ctx->viewer()->setCurrent2DImagePath("");
    m_ctx->viewer()->setImageList(QStringList(), -1);

    emit progressUpdated(40);
    m_ctx->scene()->clear3DModel();
    m_ctx->scene()->clearPointCloud();
    m_ctx->scene()->clear2DTexture();
    m_ctx->scene()->resetToSingleRenderer();

    emit progressUpdated(60);
    // Delegate to ISceneService which owns the DICOM pipeline
    m_ctx->scene()->onLoadDicom(directoryPath);

    emit loadingFinished();
    m_ctx->updateMenuStates();
}
