#include "SceneService.h"
#include "ISettingsService.h"
#include "DicomLoader.h"
#include "Model3DLoader.h"
#include "Image2DLoader.h"
#include "PanStyle.h"
#include "SignalBus.h"
#include "UserManager.h"

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
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkActor.h>
#include <algorithm>
#include <limits>
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
    // Resetting from just the model keeps a large decorative grid from shifting
    // the subject away from the centre of the viewport.
    const double max = std::numeric_limits<double>::max();
    double bounds[6] = { max, -max, max, -max, max, -max };
    for (const auto &actor : m_modelActors) {
        double actorBounds[6];
        actor->GetBounds(actorBounds);
        for (int i = 0; i < 3; ++i) {
            bounds[2 * i] = std::min(bounds[2 * i], actorBounds[2 * i]);
            bounds[2 * i + 1] = std::max(bounds[2 * i + 1], actorBounds[2 * i + 1]);
        }
    }
    if (!m_modelActors.empty()) m_renderer->ResetCamera(bounds);
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

    // The orientation marker uses window-normalised coordinates.  Keep it in
    // the lower-left corner of the DICOM 3D quadrant rather than the whole UI.
    if (m_axesWidget) m_axesWidget->SetViewport(0.51, 0.01, 0.61, 0.13);

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

void SceneService::applyViewSettings(const QString& username) {
    if (!m_renderer || !m_vtkWidget || !m_vtkWidget->renderWindow()) return;

    auto* um = UserManager::instance();
    if (!um) return;

    // ── 1. Background color ─────────────────────────────────────────────────
    int bgColorIdx = um->getUserPref(username, "view_bg_color", "0").toInt();
    switch (bgColorIdx) {
    case 1: // Light
        m_renderer->SetBackground(0.85, 0.87, 0.90);
        m_renderer->SetBackground2(1.0, 1.0, 1.0);
        m_renderer->GradientBackgroundOn();
        break;
    case 2: // Gray
        m_renderer->SetBackground(0.45, 0.45, 0.48);
        m_renderer->SetBackground2(0.60, 0.60, 0.63);
        m_renderer->GradientBackgroundOn();
        break;
    default: // Dark (0)
        m_renderer->SetBackground(0.06, 0.06, 0.08);
        m_renderer->SetBackground2(0.14, 0.14, 0.18);
        m_renderer->GradientBackgroundOn();
        break;
    }

    // 2. Axes
    bool showAxes = um->getUserPref(username, "view_show_axes", "true") == "true";
    if (showAxes) {
        if (!m_axesWidget) {
            auto axesActor = vtkSmartPointer<vtkAxesActor>::New();
            m_axesWidget = vtkOrientationMarkerWidget::New();
            m_axesWidget->SetOutlineColor(0.93, 0.57, 0.13);
            m_axesWidget->SetOrientationMarker(axesActor);
            m_axesWidget->SetInteractor(m_vtkWidget->renderWindow()->GetInteractor());
            const bool dicomLayout = m_axialRenderer && m_sagittalRenderer && m_coronalRenderer;
            m_axesWidget->SetViewport(dicomLayout ? 0.51 : 0.02, 0.01,
                                      dicomLayout ? 0.61 : 0.18, dicomLayout ? 0.13 : 0.22);
            m_axesWidget->EnabledOn();
            m_axesWidget->InteractiveOff();
        } else {
            m_axesWidget->EnabledOn();
        }
    } else {
        if (m_axesWidget) m_axesWidget->EnabledOff();
    }

    // 3. Grid
    bool showGrid = um->getUserPref(username, "view_show_grid", "true") == "true";
    if (showGrid && !m_gridActor) {
        // Build a large grid-plane slightly below the scene origin
        vtkNew<vtkPlaneSource> plane;
        plane->SetXResolution(20);
        plane->SetYResolution(20);
        plane->SetOrigin(-10.0, -10.0, -0.01);
        plane->SetPoint1( 10.0, -10.0, -0.01);
        plane->SetPoint2(-10.0,  10.0, -0.01);
        plane->Update();

        vtkNew<vtkPolyDataMapper> gridMapper;
        gridMapper->SetInputConnection(plane->GetOutputPort());

        m_gridActor = vtkSmartPointer<vtkActor>::New();
        m_gridActor->SetMapper(gridMapper);
        m_gridActor->GetProperty()->SetRepresentationToWireframe();
        m_gridActor->GetProperty()->SetColor(0.35, 0.35, 0.38);
        m_gridActor->GetProperty()->SetOpacity(0.5);
        m_gridActor->GetProperty()->LightingOff();
        m_renderer->AddActor(m_gridActor);
    } else if (!showGrid && m_gridActor) {
        m_renderer->RemoveActor(m_gridActor);
        m_gridActor = nullptr;
    }

    m_vtkWidget->renderWindow()->Render();
}
