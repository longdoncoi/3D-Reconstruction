#include "SceneService.h"
#include "ISettingsService.h"
#include "DicomLoader.h"
#include "Model3DLoader.h"
#include "Image2DLoader.h"
#include "PanStyle.h"
#include "SignalBus.h"
#include <QFileDialog>
#include "../utils/ModernMessageBox.h"
#include <vtkRenderWindow.h>
#include <vtkLight.h>
#include <vtkCornerAnnotation.h>
#include <vtkTextProperty.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkCoordinate.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkActor2D.h>
#include <vtkProperty2D.h>

SceneService::SceneService(IAppContext* ctx, QVTKOpenGLNativeWidget* vtkWidget, QObject* parent) 
    : QObject(parent), m_ctx(ctx), m_vtkWidget(vtkWidget) 
{
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_renderer->SetBackground(0, 0, 0);
    m_vtkWidget->renderWindow()->AddRenderer(m_renderer);

    vtkSmartPointer<vtkLight> headlight = vtkSmartPointer<vtkLight>::New();
    headlight->SetLightTypeToHeadlight();
    headlight->SetIntensity(1.5);
    m_renderer->AddLight(headlight);
}

SceneService::~SceneService() {
    if (m_crosshair) {
        m_crosshair->cleanup();
        delete m_crosshair;
    }
}

void SceneService::setTextureActor(vtkSmartPointer<vtkActor> actor) {
    clear2DTexture();
    m_texturePlaneActor = actor;
    if (m_texturePlaneActor) {
        m_renderer->AddActor(m_texturePlaneActor);
    }
}

void SceneService::setPointCloudActor(vtkSmartPointer<vtkActor> actor) {
    clearPointCloud();
    m_cloudActor = actor;
    if (m_cloudActor) {
        m_renderer->AddActor(m_cloudActor);
        m_pointCloudVisible = true;
    }
}

void SceneService::clear3DModel() {
    for (auto &actor : m_modelActors) {
        m_renderer->RemoveActor(actor);
    }
    m_modelActors.clear();
}

void SceneService::clear2DTexture() {
    if (m_texturePlaneActor) {
        m_renderer->RemoveActor(m_texturePlaneActor);
        m_texturePlaneActor = nullptr;
    }
}

void SceneService::clearPointCloud() {
    if (m_cloudActor) {
        m_renderer->RemoveActor(m_cloudActor);
        m_cloudActor = nullptr;
    }
    m_pointCloudVisible = false;
}

void SceneService::resetToSingleRenderer() {
    if (m_crosshair) {
        m_crosshair->cleanup();
        delete m_crosshair;
        m_crosshair = nullptr;
    }
    m_crosshairStyle = nullptr;
    auto *rw = m_vtkWidget->renderWindow();
    
    if (m_axialRenderer) { m_axialRenderer->RemoveAllViewProps(); rw->RemoveRenderer(m_axialRenderer); m_axialRenderer = nullptr; }
    if (m_sagittalRenderer) { m_sagittalRenderer->RemoveAllViewProps(); rw->RemoveRenderer(m_sagittalRenderer); m_sagittalRenderer = nullptr; }
    if (m_coronalRenderer) { m_coronalRenderer->RemoveAllViewProps(); rw->RemoveRenderer(m_coronalRenderer); m_coronalRenderer = nullptr; }
    
    m_renderer->SetViewport(0.0, 0.0, 1.0, 1.0);
    m_renderer->RemoveAllViewProps();
    
    vtkNew<PanStyle> style;
    if (rw->GetInteractor()) rw->GetInteractor()->SetInteractorStyle(style);
    
    m_renderer->SetBackground(0, 0, 0);
    rw->Render();
}

void SceneService::loadOBJwithMTL(const QString &objPath, const QString &mtlPath) {
    clear3DModel();
    auto actors = Model3DLoader::load(objPath, mtlPath);
    for (auto &a : actors) {
        m_modelActors.push_back(a);
        m_renderer->AddActor(a);
    }
    m_renderer->ResetCamera();
    m_vtkWidget->renderWindow()->Render();
}

void SceneService::onLoadDicom(const QString &path) {
    QString fn = path;
    if (fn.isEmpty()) {
        fn = QFileDialog::getExistingDirectory(m_ctx->mainWindow(), m_ctx->translate("file.select_dicom"), m_ctx->settings()->getLastUsedPath("viewer_dicom"), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    }
    if (fn.isEmpty()) return;

    m_ctx->settings()->setLastUsedPath("viewer_dicom", fn);
    auto volume = DicomLoader::loadSeries(fn);
    if (!volume || volume->GetNumberOfPoints() < 1) {
        ModernMessageBox::warning(m_ctx->mainWindow(), m_ctx->translate("dicom.load_err_title"), m_ctx->translate("dicom.load_err_msg"));
        return;
    }

    clear3DModel();
    clear2DTexture();
    clearPointCloud();
    m_renderer->RemoveAllViewProps();

    setupDicomRenderers(volume);
    setupCrosshairInteractor();

    m_axialRenderer->ResetCamera();
    m_sagittalRenderer->ResetCamera();
    m_coronalRenderer->ResetCamera();
    m_renderer->ResetCamera();
    
    m_vtkWidget->renderWindow()->Render();
}

void SceneService::setupDicomRenderers(vtkSmartPointer<vtkImageData> volume) {
    if (!m_axialRenderer) m_axialRenderer = vtkSmartPointer<vtkRenderer>::New();
    if (!m_sagittalRenderer) m_sagittalRenderer = vtkSmartPointer<vtkRenderer>::New();
    if (!m_coronalRenderer) m_coronalRenderer = vtkSmartPointer<vtkRenderer>::New();

    m_axialRenderer->SetViewport(0.0, 0.5, 0.5, 1.0);
    m_sagittalRenderer->SetViewport(0.5, 0.5, 1.0, 1.0);
    m_coronalRenderer->SetViewport(0.0, 0.0, 0.5, 0.5);
    m_renderer->SetViewport(0.5, 0.0, 1.0, 0.5);

    m_axialRenderer->SetBackground(0, 0, 0);
    m_sagittalRenderer->SetBackground(0, 0, 0);
    m_coronalRenderer->SetBackground(0, 0, 0);
    m_renderer->SetBackground(0, 0, 0);

    auto *rw = m_vtkWidget->renderWindow();
    rw->AddRenderer(m_axialRenderer);
    rw->AddRenderer(m_sagittalRenderer);
    rw->AddRenderer(m_coronalRenderer);

    auto addTitle = [](vtkRenderer* ren, const char* text) {
        vtkNew<vtkCornerAnnotation> ann;
        ann->SetText(2, text);
        ann->GetTextProperty()->SetColor(0.0, 1.0, 0.0);
        ann->GetTextProperty()->SetFontSize(16);
        ren->AddViewProp(ann);
    };

    addTitle(m_axialRenderer, "Axial");
    addTitle(m_sagittalRenderer, "Sagittal");
    addTitle(m_coronalRenderer, "Coronal");
    addTitle(m_renderer, "3D View");

    auto addBorder = [](vtkRenderer* ren) {
        vtkNew<vtkPoints> points;
        points->InsertNextPoint(0.0, 0.0, 0.0);
        points->InsertNextPoint(1.0, 0.0, 0.0);
        points->InsertNextPoint(1.0, 1.0, 0.0);
        points->InsertNextPoint(0.0, 1.0, 0.0);
        points->InsertNextPoint(0.0, 0.0, 0.0);

        vtkNew<vtkCellArray> lines;
        vtkIdType lineIds[5] = {0, 1, 2, 3, 4};
        lines->InsertNextCell(5, lineIds);

        vtkNew<vtkPolyData> polyData;
        polyData->SetPoints(points);
        polyData->SetLines(lines);

        vtkNew<vtkCoordinate> coord;
        coord->SetCoordinateSystemToNormalizedViewport();

        vtkNew<vtkPolyDataMapper2D> mapper;
        mapper->SetInputData(polyData);
        mapper->SetTransformCoordinate(coord);

        vtkNew<vtkActor2D> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(0.25, 0.25, 0.3); // Sleek modern slate-grey line
        actor->GetProperty()->SetLineWidth(2.0);

        ren->AddViewProp(actor);
    };

    addBorder(m_axialRenderer);
    addBorder(m_sagittalRenderer);
    addBorder(m_coronalRenderer);
    addBorder(m_renderer);

    double r[2];
    volume->GetScalarRange(r);
    m_renderer->AddViewProp(DicomLoader::createVolume(volume, r));

    if (m_crosshair) {
        m_crosshair->cleanup();
        delete m_crosshair;
    }
    m_crosshair = new CrosshairManager(this);
    m_crosshair->initialize(volume, m_sagittalRenderer, m_coronalRenderer, m_axialRenderer, rw);
}

void SceneService::setupCrosshairInteractor() {
    auto style = vtkSmartPointer<CrosshairInteractorStyle>::New();
    style->manager = m_crosshair;
    style->renderer3D = m_renderer;
    m_crosshairStyle = style;

    m_vtkWidget->renderWindow()->GetInteractor()->SetInteractorStyle(m_crosshairStyle);
}
