// ============================================================
// CrosshairManager.cpp
// ============================================================
#include "CrosshairManager.h"
#include "CrosshairGeometry.h"

#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkTextProperty.h>
#include <vtkCoordinate.h>
#include <vtkMatrix4x4.h>
#include <vtkImageProperty.h>
#include <vtkMath.h>
#include <vtkNew.h>
#include <vtkCamera.h>
#include <vtkImageMapper3D.h>
#include <vtkRenderer.h>

#include <cmath>
#include <algorithm>

vtkStandardNewMacro(CrosshairInteractorStyle);

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

CrosshairManager::CrosshairManager(QObject *parent) : QObject(parent) {}
CrosshairManager::~CrosshairManager() { cleanup(); }

void CrosshairManager::cleanup()
{
    for (int i = 0; i < 3; ++i) {
        auto &v   = m_views[i];
        vtkRenderer *ren = m_renderers[i];
        if (!ren) continue;

        if (v.imageActor)  ren->RemoveActor(v.imageActor);
        if (v.hActor)      ren->RemoveActor(v.hActor);
        if (v.vActor)      ren->RemoveActor(v.vActor);
        // anchorActor is NOT in the renderer (hidden for hit-test only)
        for (int k = 0; k < 4; ++k)
            if (v.labels[k]) ren->RemoveActor(v.labels[k]);

        v = CrosshairOverlay();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  initialize
//
//  CALLER RESPONSIBILITY (in onLoadDicom, BEFORE calling this):
//    axialRenderer->RemoveAllViewProps();
//    sagittalRenderer->RemoveAllViewProps();
//    coronalRenderer->RemoveAllViewProps();
//  This removes the actors created by DicomLoader::createSlice() so there is
//  exactly ONE imageActor per renderer (the one we create below).
// ─────────────────────────────────────────────────────────────────────────────

void CrosshairManager::initialize(vtkImageData*    volume,
                                  vtkRenderer*     sagittalRenderer,
                                  vtkRenderer*     coronalRenderer,
                                  vtkRenderer*     axialRenderer,
                                  vtkRenderWindow* renderWindow,
                                  double window, double level)
{
    m_volume       = volume;
    m_renderWindow = renderWindow;
    
    if (window < 0 || level < 0) {
        double range[2];
        volume->GetScalarRange(range);
        m_window = range[1] - range[0];
        m_level = (range[0] + range[1]) / 2.0;
    } else {
        m_window = window;
        m_level = level;
    }

    m_renderers[ORI_SAGITTAL] = sagittalRenderer;
    m_renderers[ORI_CORONAL]  = coronalRenderer;
    m_renderers[ORI_AXIAL]    = axialRenderer;

    volume->GetCenter(m_center);

    for (int i = 0; i < 3; ++i) {
        m_views[i].orientation = i;
        buildOverlay(i);
    }
    updateAllViews();
}

// ─────────────────────────────────────────────────────────────────────────────
//  buildOverlay – create ALL actors for one view (called once per view)
// ─────────────────────────────────────────────────────────────────────────────

void CrosshairManager::buildOverlay(int vi)
{
    CrosshairOverlay &v   = m_views[vi];
    vtkRenderer      *ren = m_renderers[vi];
    const double     *col = CrosshairGeometry::lineColor(vi);

    // ── 1. Reslice pipeline ──────────────────────────────────────────────────
    //    This is the ONLY imageActor added to this renderer.
    {
        v.reslice = vtkSmartPointer<vtkImageReslice>::New();
        v.reslice->SetInputData(m_volume);
        v.reslice->SetOutputDimensionality(2);
        v.reslice->SetInterpolationModeToLinear();

        vtkNew<vtkMatrix4x4> mat;
        mat->DeepCopy(CrosshairGeometry::resliceAxes(vi));
        mat->SetElement(0, 3, m_center[0]);
        mat->SetElement(1, 3, m_center[1]);
        mat->SetElement(2, 3, m_center[2]);
        v.reslice->SetResliceAxes(mat);
        v.reslice->Update();

        v.imageActor = vtkSmartPointer<vtkImageActor>::New();
        v.imageActor->GetMapper()->SetInputConnection(v.reslice->GetOutputPort());
        v.imageActor->GetProperty()->SetColorWindow(m_window);
        v.imageActor->GetProperty()->SetColorLevel(m_level);
        ren->AddActor(v.imageActor);
    }

    // ── 2. Crosshair lines (normalised-viewport 2D actors) ───────────────────
    auto makeLine2D = [&](vtkSmartPointer<vtkLineSource>       &src,
                          vtkSmartPointer<vtkPolyDataMapper2D> &mapper,
                          vtkSmartPointer<vtkActor2D>          &actor)
    {
        src    = vtkSmartPointer<vtkLineSource>::New();
        mapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
        actor  = vtkSmartPointer<vtkActor2D>::New();

        vtkNew<vtkCoordinate> coord;
        coord->SetCoordinateSystemToNormalizedViewport();
        mapper->SetInputConnection(src->GetOutputPort());
        mapper->SetTransformCoordinate(coord);
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(col[0], col[1], col[2]);
        actor->GetProperty()->SetLineWidth(1.5);
        actor->GetProperty()->SetOpacity(0.9);
        ren->AddActor(actor);
    };
    makeLine2D(v.hLine, v.hMapper, v.hActor);
    makeLine2D(v.vLine, v.vMapper, v.vActor);

    // ── 3. Anatomical labels ─────────────────────────────────────────────────
    for (int k = 0; k < 4; ++k) {
        v.labels[k] = vtkSmartPointer<vtkTextActor>::New();
        v.labels[k]->GetTextProperty()->SetFontSize(15);
        v.labels[k]->GetTextProperty()->SetColor(col[0], col[1], col[2]);
        v.labels[k]->GetTextProperty()->SetBold(1);
        v.labels[k]->GetTextProperty()->SetFontFamilyToArial();
        v.labels[k]->GetTextProperty()->SetShadow(1);
        v.labels[k]->GetPositionCoordinate()
            ->SetCoordinateSystemToNormalizedViewport();
        ren->AddActor(v.labels[k]);
    }

    // ── 4. Anchor sphere (hidden – kept only for isNearAnchor hit-testing) ────
    //    The sphere is in 3D world space, but the MPR cameras look at 2D reslice
    //    output space, so the sphere would project at a wrong/random position on
    //    screen and trigger VTK's red selection ring. We create the geometry so
    //    that isNearAnchor() can measure its screen-space position via the
    //    viewport transform, but we do NOT add it to the renderer.
    {
        v.anchorSource = vtkSmartPointer<vtkSphereSource>::New();
        v.anchorSource->SetThetaResolution(16);
        v.anchorSource->SetPhiResolution(16);

        v.anchorMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        v.anchorMapper->SetInputConnection(v.anchorSource->GetOutputPort());

        v.anchorActor = vtkSmartPointer<vtkActor>::New();
        v.anchorActor->SetMapper(v.anchorMapper);
        // Not added to renderer – invisible but used for hit-test geometry.
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Update helpers
// ─────────────────────────────────────────────────────────────────────────────

void CrosshairManager::updateAllViews()
{
    for (int i = 0; i < 3; ++i) {
        updateReslice(i);
        updateOverlay(i);
    }
    if (m_renderWindow) m_renderWindow->Render();
}

void CrosshairManager::updateReslice(int vi)
{
    auto &v = m_views[vi];
    if (!v.reslice) return;
    vtkMatrix4x4 *mat = v.reslice->GetResliceAxes();
    mat->SetElement(0, 3, m_center[0]);
    mat->SetElement(1, 3, m_center[1]);
    mat->SetElement(2, 3, m_center[2]);
    v.reslice->Update();
}

void CrosshairManager::updateOverlay(int vi)
{
    auto &v   = m_views[vi];
    vtkRenderer *ren = m_renderers[vi];
    if (!ren || !v.hLine) return;

    double bounds[6];
    m_volume->GetBounds(bounds);

    double nx = 0.5, ny = 0.5;
    CrosshairGeometry::worldToNormalized(vi, m_center, bounds, nx, ny);

    // Lines
    v.hLine->SetPoint1(0.0, ny, 0.0);  v.hLine->SetPoint2(1.0, ny, 0.0);
    v.hLine->Update();
    v.vLine->SetPoint1(nx, 0.0, 0.0);  v.vLine->SetPoint2(nx, 1.0, 0.0);
    v.vLine->Update();

    // Labels
    const char* lbl[4];
    getLabels(vi, lbl);
    const double pad = 0.025;

    v.labels[0]->SetInput(lbl[0]);
    v.labels[0]->GetPositionCoordinate()->SetValue(pad, ny + 0.01);

    v.labels[1]->SetInput(lbl[1]);
    v.labels[1]->GetPositionCoordinate()->SetValue(1.0 - pad - 0.05, ny + 0.01);

    v.labels[2]->SetInput(lbl[2]);
    v.labels[2]->GetPositionCoordinate()->SetValue(nx + 0.01, 1.0 - pad - 0.06);

    v.labels[3]->SetInput(lbl[3]);
    v.labels[3]->GetPositionCoordinate()->SetValue(nx + 0.01, pad);

    // Anchor sphere
    double sp[6]; m_volume->GetBounds(sp);
    double dim = std::max({sp[1]-sp[0], sp[3]-sp[2], sp[5]-sp[4]});
    v.anchorSource->SetRadius(dim * 0.012);
    v.anchorSource->SetCenter(m_center[0], m_center[1], m_center[2]);
    v.anchorSource->Update();
}

void CrosshairManager::getLabels(int vi, const char* out[4])
{
    CrosshairGeometry::labels(vi, out);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Anchor hit-test & coord conversion
// ─────────────────────────────────────────────────────────────────────────────

bool CrosshairManager::isNearAnchor(int vi, int x, int y)
{
    vtkRenderer *ren = m_renderers[vi];
    if (!ren || !m_renderWindow) return false;

    double bounds[6]; m_volume->GetBounds(bounds);
    double nx, ny;
    CrosshairGeometry::worldToNormalized(vi, m_center, bounds, nx, ny);

    double vp[4]; ren->GetViewport(vp);
    int *ws = m_renderWindow->GetSize();
    double ax = (vp[0] + nx*(vp[2]-vp[0])) * ws[0];
    double ay = (vp[1] + ny*(vp[3]-vp[1])) * ws[1];

    return std::sqrt((x-ax)*(x-ax)+(y-ay)*(y-ay)) < 20.0;
}

bool CrosshairManager::displayToWorld(int vi, int x, int y, double worldPt[3])
{
    vtkRenderer *ren = m_renderers[vi];
    if (!ren || !m_renderWindow) return false;

    double vp[4]; ren->GetViewport(vp);
    int *ws = m_renderWindow->GetSize();
    double nx = ((double)x/ws[0]-vp[0]) / (vp[2]-vp[0]);
    double ny = ((double)y/ws[1]-vp[1]) / (vp[3]-vp[1]);
    nx = std::max(0.0,std::min(1.0,nx));
    ny = std::max(0.0,std::min(1.0,ny));

    double bounds[6]; m_volume->GetBounds(bounds);
    CrosshairGeometry::normalizedToWorld(vi, nx, ny, bounds, m_center, worldPt);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mouse handlers
// ─────────────────────────────────────────────────────────────────────────────

void CrosshairManager::onLeftButtonDown(int vi, int x, int y)
{
    if (vi < 0 || vi > 2) return;
    m_views[vi].dragging = isNearAnchor(vi, x, y);
}

void CrosshairManager::onMouseMove(int vi, int x, int y)
{
    if (vi < 0 || vi > 2 || !m_views[vi].dragging) return;

    double newWorld[3];
    if (displayToWorld(vi, x, y, newWorld)) {
        m_center[0] = newWorld[0];
        m_center[1] = newWorld[1];
        m_center[2] = newWorld[2];

        // Re-slice the OTHER two views; just move crosshair in the dragged view
        for (int i = 0; i < 3; ++i) {
            if (i != vi) updateReslice(i);
            updateOverlay(i);
        }
        if (m_renderWindow) m_renderWindow->Render();
    }
}

void CrosshairManager::onLeftButtonUp(int vi)
{
    if (vi >= 0 && vi < 3) m_views[vi].dragging = false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CrosshairInteractorStyle
// ─────────────────────────────────────────────────────────────────────────────

int CrosshairInteractorStyle::detectView() const
{
    if (!manager || !Interactor) return -1;
    int x = Interactor->GetEventPosition()[0];
    int y = Interactor->GetEventPosition()[1];
    int *ws = Interactor->GetRenderWindow()->GetSize();
    double fx = (double)x/ws[0], fy = (double)y/ws[1];

    for (int i = 0; i < 3; ++i) {
        vtkRenderer *ren = manager->renderer(i);
        if (!ren) continue;
        double vp[4]; ren->GetViewport(vp);
        if (fx>=vp[0] && fx<=vp[2] && fy>=vp[1] && fy<=vp[3]) return i;
    }
    return -1;
}

bool CrosshairInteractorStyle::isIn3DView(int x, int y) const
{
    if (!renderer3D || !Interactor) return false;
    int *ws = Interactor->GetRenderWindow()->GetSize();
    double fx = (double)x/ws[0], fy = (double)y/ws[1];
    double vp[4]; renderer3D->GetViewport(vp);
    return fx >= vp[0] && fx <= vp[2] && fy >= vp[1] && fy <= vp[3];
}

void CrosshairInteractorStyle::OnLeftButtonDown()
{
    int x = Interactor->GetEventPosition()[0];
    int y = Interactor->GetEventPosition()[1];

    if (manager) {
        m_activeView = detectView();
        if (m_activeView >= 0) {
            this->CurrentRenderer = manager->renderer(m_activeView);
            manager->onLeftButtonDown(m_activeView, x, y);
            if (manager->isDragging(m_activeView)) return; // consumed
        }
    }

    // Left-drag in the 3D viewport starts rotation
    if (isIn3DView(x, y)) {
        if (renderer3D) this->CurrentRenderer = renderer3D;
        m_rotating3D = true;
        m_lastX = x;
        m_lastY = y;
        return;
    }

    // Otherwise, fallback to default behavior only if NOT in an MPR view
    if (m_activeView < 0) {
        Superclass::OnLeftButtonDown();
    }
}

void CrosshairInteractorStyle::OnMouseMove()
{
    int x = Interactor->GetEventPosition()[0];
    int y = Interactor->GetEventPosition()[1];

    // Crosshair drag in MPR view
    if (manager && m_activeView >= 0 && manager->isDragging(m_activeView)) {
        manager->onMouseMove(m_activeView, x, y);
        return; // consumed
    }

    // Trackball rotation in 3D view
    if (m_rotating3D && renderer3D) {
        int dx = x - m_lastX;
        int dy = y - m_lastY;
        m_lastX = x;
        m_lastY = y;

        int *sz = Interactor->GetRenderWindow()->GetSize();
        double azimuth   = -(double)dx / sz[0] * 360.0;
        double elevation =  (double)dy / sz[1] * 360.0;

        vtkCamera *cam = renderer3D->GetActiveCamera();
        cam->Azimuth(azimuth);
        cam->Elevation(elevation);
        cam->OrthogonalizeViewUp();
        renderer3D->ResetCameraClippingRange();
        Interactor->GetRenderWindow()->Render();
        return;
    }

    // Default behavior (Window/Level or Pan/Dolly) only if NOT in an MPR viewport or explicitly allowed
    if (m_activeView < 0 && !isIn3DView(x, y)) {
        if (this->State == VTKIS_PAN || this->State == VTKIS_DOLLY) {
            Superclass::OnMouseMove();
        }
        // Superclass::OnMouseMove(); // Disable default Window/Level adjustment
    } else {
        Superclass::OnMouseMove();
    }
}

void CrosshairInteractorStyle::OnLeftButtonUp()
{
    if (manager && m_activeView >= 0) {
        manager->onLeftButtonUp(m_activeView);
        m_activeView = -1;
        return;
    }

    if (m_rotating3D) {
        m_rotating3D = false;
        return;
    }

    Superclass::OnLeftButtonUp();
}

void CrosshairInteractorStyle::OnRightButtonDown()
{
    m_activeView = detectView();
    if (m_activeView >= 0) {
        this->CurrentRenderer = manager->renderer(m_activeView);
    } else if (renderer3D && isIn3DView(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1])) {
        this->CurrentRenderer = renderer3D;
    }
    this->StartPan();
}

void CrosshairInteractorStyle::OnRightButtonUp()
{
    this->EndPan();
    if (this->State == VTKIS_NONE) {
        m_activeView = -1;
    }
}

void CrosshairInteractorStyle::OnMiddleButtonDown()
{
    m_activeView = detectView();
    if (m_activeView >= 0) {
        this->CurrentRenderer = manager->renderer(m_activeView);
    } else if (renderer3D && isIn3DView(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1])) {
        this->CurrentRenderer = renderer3D;
    }
    this->StartDolly();
}

void CrosshairInteractorStyle::OnMiddleButtonUp()
{
    this->EndDolly();
    if (this->State == VTKIS_NONE) {
        m_activeView = -1;
    }
}
