#include <QtTest>

#include "CrosshairManager.h"
#include "CrosshairGeometry.h"

#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

/**
 * @brief Contract of the MPR crosshair overlay (CrosshairManager).
 *
 * The manager owns VTK actors that live in the *shell's* renderers, so its
 * lifecycle is the interesting part: initialize() must install the overlay in
 * all three views, cleanup() must take it out again, and the drag state
 * machine must only start when the pointer is really on the anchor.
 *
 * The interactor-style contract is the second half: one style is shared by the
 * render window, detects the viewport under the pointer and routes a left drag
 * to the crosshair, the 3-D camera or the VTK default. The test drives that
 * state machine directly with synthetic event positions.
 *
 * The test never opens a GL context: the renderers and the window are only
 * used for their geometry (viewport, size), which is exactly what the
 * hit-testing works on.
 */
class TestCrosshairManager : public QObject {
    Q_OBJECT

private slots:
    void initializeBuildsTheOverlayInEveryView();
    void cleanupRemovesEveryActorAgain();
    void draggingStartsOnlyNearTheAnchor();
    void mouseMoveWithoutDragIsIgnored();
    void invalidViewIndexIsIgnored();
    void interactorStartsDragInEachMprViewport();
    void interactorRotatesIn3DViewport();
    void interactorPansAndDollies();
    void interactorFallsBackWhenNoViewMatches();
};

namespace {

/// A small synthetic volume with a non-degenerate scalar range.
vtkSmartPointer<vtkImageData> makeVolume(int size = 16)
{
    vtkSmartPointer<vtkImageData> volume = vtkSmartPointer<vtkImageData>::New();
    volume->SetDimensions(size, size, size);
    volume->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    for (int z = 0; z < size; ++z) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                if (auto *scalar = static_cast<unsigned char *>(volume->GetScalarPointer(x, y, z))) {
                    scalar[0] = static_cast<unsigned char>((x + y + z) % 251);
                }
            }
        }
    }
    return volume;
}

/// The overlay is made of 2-D actors and text actors: VTK keeps them in the
/// view-prop collection, the 3-D anchor in the actor collection.  The overlay
/// is "installed" when both together carry something.
int propsIn(vtkRenderer *renderer)
{
    return renderer->GetActors()->GetNumberOfItems()
         + renderer->GetViewProps()->GetNumberOfItems();
}

/// A RenderWindow is only needed for its size and for the interactor's event
/// plumbing. Overriding Render() keeps the rotation path (which asks the
/// window to repaint) away from an OpenGL context the test does not have.
class HeadlessRenderWindow : public vtkRenderWindow {
public:
    static HeadlessRenderWindow *New();
    vtkTypeMacro(HeadlessRenderWindow, vtkRenderWindow);
    void Render() override {}
};
vtkStandardNewMacro(HeadlessRenderWindow);

/// The 2×2 MPR + 3-D layout used by SceneService: the three MPR viewports
/// tile the left/top halves and the 3-D view occupies the bottom-right.
/// A fixture is built per test because a drag mutates the shared anchor.
class MprFixture {
public:
    MprFixture()
    {
        window->SetSize(300, 300);
        window->AddRenderer(sagittal);
        window->AddRenderer(coronal);
        window->AddRenderer(axial);
        window->AddRenderer(threeD);

        sagittal->SetViewport(0.0, 0.5, 0.5, 1.0);
        coronal->SetViewport(0.5, 0.5, 1.0, 1.0);
        axial->SetViewport(0.0, 0.0, 0.5, 0.5);
        threeD->SetViewport(0.5, 0.0, 1.0, 0.5);

        manager.initialize(volume, sagittal, coronal, axial, window);

        style->manager = &manager;
        style->renderer3D = threeD;
        style->SetInteractor(interactor);
        interactor->SetRenderWindow(window);
    }

    void moveTo(int x, int y) { interactor->SetEventPosition(x, y); }

    vtkSmartPointer<vtkImageData> volume = makeVolume();
    vtkSmartPointer<vtkRenderer> sagittal = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderer> coronal = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderer> axial = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderer> threeD = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<HeadlessRenderWindow> window = vtkSmartPointer<HeadlessRenderWindow>::New();
    vtkSmartPointer<vtkRenderWindowInteractor> interactor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    vtkSmartPointer<CrosshairInteractorStyle> style =
        vtkSmartPointer<CrosshairInteractorStyle>::New();
    CrosshairManager manager;
};

} // namespace

void TestCrosshairManager::initializeBuildsTheOverlayInEveryView()
{
    const auto volume = makeVolume();
    vtkNew<vtkRenderer> sagittal;
    vtkNew<vtkRenderer> coronal;
    vtkNew<vtkRenderer> axial;
    vtkNew<vtkRenderWindow> window;
    window->SetSize(300, 300);

    CrosshairManager manager;
    manager.initialize(volume, sagittal, coronal, axial, window);

    for (int i = 0; i < 3; ++i) {
        QVERIFY2(manager.renderer(i) != nullptr,
                 qPrintable(QStringLiteral("no renderer for view %1").arg(i)));
        const CrosshairOverlay &view = manager.m_views[i];
        QVERIFY(view.hActor != nullptr);
        QVERIFY(view.vActor != nullptr);
        QVERIFY(view.anchorActor != nullptr);
        QVERIFY(view.imageActor != nullptr);
        QVERIFY(view.reslice != nullptr);
        for (int k = 0; k < 4; ++k) {
            QVERIFY2(view.labels[k] != nullptr,
                     qPrintable(QStringLiteral("missing label %1 in view %2").arg(k).arg(i)));
        }
        QVERIFY(!manager.isDragging(i));
    }
    QVERIFY(!manager.anyDragging());

    // The overlay must actually be installed in the shell's renderers.
    for (vtkRenderer *renderer : {sagittal.Get(), coronal.Get(), axial.Get()}) {
        QVERIFY2(propsIn(renderer) > 0,
                 "the overlay must be installed in the shell's renderer");
    }
}

void TestCrosshairManager::cleanupRemovesEveryActorAgain()
{
    const auto volume = makeVolume();
    vtkNew<vtkRenderer> sagittal;
    vtkNew<vtkRenderer> coronal;
    vtkNew<vtkRenderer> axial;
    vtkNew<vtkRenderWindow> window;
    window->SetSize(300, 300);

    CrosshairManager manager;
    manager.initialize(volume, sagittal, coronal, axial, window);

    manager.cleanup();
    for (vtkRenderer *renderer : {sagittal.Get(), coronal.Get(), axial.Get()}) {
        QCOMPARE(propsIn(renderer), 0);
    }
    // The overlay state must be reset as well — a second cleanup() is a no-op.
    manager.cleanup();
    for (int i = 0; i < 3; ++i) {
        QVERIFY(manager.m_views[i].hActor == nullptr);
        QVERIFY(manager.m_views[i].imageActor == nullptr);
    }
}

void TestCrosshairManager::draggingStartsOnlyNearTheAnchor()
{
    const auto volume = makeVolume();
    vtkNew<vtkRenderer> sagittal;
    vtkNew<vtkRenderer> coronal;
    vtkNew<vtkRenderer> axial;
    vtkNew<vtkRenderWindow> window;
    window->SetSize(300, 300);

    CrosshairManager manager;
    manager.initialize(volume, sagittal, coronal, axial, window);

    // The anchor sits at the volume centre → the middle of the viewport.
    manager.onLeftButtonDown(ORI_AXIAL, 150, 150);
    QVERIFY(manager.isDragging(ORI_AXIAL));
    QVERIFY(manager.anyDragging());

    manager.onMouseMove(ORI_AXIAL, 160, 160);
    manager.onMouseMove(ORI_SAGITTAL, 160, 160);
    manager.onMouseMove(ORI_CORONAL, 160, 160);

    manager.onLeftButtonUp(ORI_AXIAL);
    QVERIFY(!manager.isDragging(ORI_AXIAL));
    QVERIFY(!manager.anyDragging());

    // A click far away from the anchor must not start a drag.
    manager.onLeftButtonDown(ORI_AXIAL, 5, 5);
    QVERIFY(!manager.isDragging(ORI_AXIAL));
}

void TestCrosshairManager::mouseMoveWithoutDragIsIgnored()
{
    const auto volume = makeVolume();
    vtkNew<vtkRenderer> sagittal;
    vtkNew<vtkRenderer> coronal;
    vtkNew<vtkRenderer> axial;
    vtkNew<vtkRenderWindow> window;
    window->SetSize(300, 300);

    CrosshairManager manager;
    manager.initialize(volume, sagittal, coronal, axial, window);

    // No drag started → the move must not change anything and must not crash.
    manager.onMouseMove(ORI_AXIAL, 200, 200);
    QVERIFY(!manager.anyDragging());
}

void TestCrosshairManager::invalidViewIndexIsIgnored()
{
    const auto volume = makeVolume();
    vtkNew<vtkRenderer> sagittal;
    vtkNew<vtkRenderer> coronal;
    vtkNew<vtkRenderer> axial;
    vtkNew<vtkRenderWindow> window;

    CrosshairManager manager;
    manager.initialize(volume, sagittal, coronal, axial, window);

    manager.onLeftButtonDown(-1, 10, 10);
    manager.onLeftButtonDown(3, 10, 10);
    manager.onMouseMove(9, 10, 10);
    manager.onLeftButtonUp(-5);
    QVERIFY(!manager.anyDragging());
    QVERIFY(manager.renderer(-1) == nullptr);
    QVERIFY(manager.renderer(3) == nullptr);
}

void TestCrosshairManager::interactorStartsDragInEachMprViewport()
{
    // Each viewport owns the anchor at a different pixel; 75/225 are the
    // quarter boundaries of the 300×300 window.
    struct Case { int view; int x; int y; };
    const Case cases[] = {
        {ORI_SAGITTAL, 75, 225},
        {ORI_CORONAL, 225, 225},
        {ORI_AXIAL,    75,  75},
    };

    for (const Case &c : cases) {
        MprFixture f;

        if (c.view == ORI_AXIAL) {
            const double before = f.manager.m_views[ORI_AXIAL].hLine->GetPoint1()[1];
            f.moveTo(c.x, c.y);
            f.style->OnLeftButtonDown();
            QVERIFY2(f.manager.isDragging(c.view),
                     qPrintable(QStringLiteral("no drag started in view %1").arg(c.view)));

            f.moveTo(c.x + 30, c.y + 30);
            f.style->OnMouseMove();
            QVERIFY(f.manager.isDragging(c.view));
            const double after = f.manager.m_views[ORI_AXIAL].hLine->GetPoint1()[1];
            QVERIFY2(std::abs(after - before) > 0.01,
                     "the drag must move the crosshair overlay, not only the flag");
        } else {
            f.moveTo(c.x, c.y);
            f.style->OnLeftButtonDown();
            QVERIFY2(f.manager.isDragging(c.view),
                     qPrintable(QStringLiteral("no drag started in view %1").arg(c.view)));
        }

        f.style->OnLeftButtonUp();
        QVERIFY(!f.manager.isDragging(c.view));
        QVERIFY(!f.manager.anyDragging());
    }
}

void TestCrosshairManager::interactorRotatesIn3DViewport()
{
    MprFixture f;

    double before[3];
    f.threeD->GetActiveCamera()->GetPosition(before);

    // Bottom-right quadrant is the 3-D viewport: no MPR renderer matches, so
    // the style starts a trackball rotation instead of a crosshair drag.
    f.moveTo(225, 75);
    f.style->OnLeftButtonDown();
    QVERIFY(!f.manager.anyDragging());

    f.moveTo(255, 105);
    f.style->OnMouseMove();

    double after[3];
    f.threeD->GetActiveCamera()->GetPosition(after);
    const double moved = std::abs(after[0] - before[0])
                       + std::abs(after[1] - before[1])
                       + std::abs(after[2] - before[2]);
    QVERIFY2(moved > 1e-6, "a left drag in the 3-D viewport must rotate the camera");

    f.style->OnLeftButtonUp();
    QCOMPARE(f.style->GetState(), VTKIS_NONE);
}

void TestCrosshairManager::interactorPansAndDollies()
{
    MprFixture f;

    // Right drag inside an MPR viewport → pan, with that renderer as target.
    f.moveTo(75, 75);
    f.style->OnRightButtonDown();
    QCOMPARE(f.style->GetState(), VTKIS_PAN);
    f.moveTo(90, 90);
    f.style->OnMouseMove();
    f.style->OnRightButtonUp();
    QCOMPARE(f.style->GetState(), VTKIS_NONE);

    // Right drag inside the 3-D viewport → pan against the 3-D renderer.
    f.moveTo(225, 75);
    f.style->OnRightButtonDown();
    QCOMPARE(f.style->GetState(), VTKIS_PAN);
    f.moveTo(255, 105);
    f.style->OnMouseMove();
    f.style->OnRightButtonUp();
    QCOMPARE(f.style->GetState(), VTKIS_NONE);

    // Middle drag inside an MPR viewport → dolly against that renderer.
    f.moveTo(75, 75);
    f.style->OnMiddleButtonDown();
    QCOMPARE(f.style->GetState(), VTKIS_DOLLY);
    f.moveTo(90, 90);
    f.style->OnMouseMove();
    f.style->OnMiddleButtonUp();
    QCOMPARE(f.style->GetState(), VTKIS_NONE);

    // Middle drag inside the 3-D viewport → dolly, with the 3-D renderer.
    f.moveTo(225, 75);
    f.style->OnMiddleButtonDown();
    QCOMPARE(f.style->GetState(), VTKIS_DOLLY);
    f.moveTo(255, 105);
    f.style->OnMouseMove();
    f.style->OnMiddleButtonUp();
    QCOMPARE(f.style->GetState(), VTKIS_NONE);
}

void TestCrosshairManager::interactorFallsBackWhenNoViewMatches()
{
    // A manager that was never initialized has three null renderers: detectView
    // must skip them (not crash), and with no 3-D renderer the style must fall
    // back to VTK's default handling.
    vtkNew<HeadlessRenderWindow> window;
    window->SetSize(300, 300);
    vtkNew<vtkRenderer> viewer;
    window->AddRenderer(viewer);

    vtkNew<vtkRenderWindowInteractor> interactor;
    interactor->SetRenderWindow(window);

    CrosshairManager uninitialized;
    vtkNew<CrosshairInteractorStyle> style;
    style->manager = &uninitialized;
    style->renderer3D = nullptr;
    style->SetInteractor(interactor);

    interactor->SetEventPosition(150, 150);
    style->OnLeftButtonDown();
    style->OnMouseMove();
    style->OnLeftButtonUp();
    QVERIFY(!uninitialized.anyDragging());

    // No active view + VTKIS_NONE → the move is a no-op, not a pan.
    interactor->SetEventPosition(150, 150);
    style->OnMouseMove();
    QCOMPARE(style->GetState(), VTKIS_NONE);

    // With no viewport under the pointer, pan/dolly still use VTK's defaults.
    style->OnRightButtonDown();
    QCOMPARE(style->GetState(), VTKIS_PAN);
    style->OnMouseMove();
    style->OnRightButtonUp();
    QCOMPARE(style->GetState(), VTKIS_NONE);

    style->OnMiddleButtonDown();
    QCOMPARE(style->GetState(), VTKIS_DOLLY);
    style->OnMouseMove();
    style->OnMiddleButtonUp();
    QCOMPARE(style->GetState(), VTKIS_NONE);
}

QTEST_GUILESS_MAIN(TestCrosshairManager)
#include "TestCrosshairManager.moc"
