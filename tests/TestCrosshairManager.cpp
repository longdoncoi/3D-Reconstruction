#include <QtTest>

#include "CrosshairManager.h"
#include "CrosshairGeometry.h"

#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

/**
 * @brief Contract of the MPR crosshair overlay (CrosshairManager).
 *
 * The manager owns VTK actors that live in the *shell's* renderers, so its
 * lifecycle is the interesting part: initialize() must install the overlay in
 * all three views, cleanup() must take it out again, and the drag state
 * machine must only start when the pointer is really on the anchor.
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

QTEST_GUILESS_MAIN(TestCrosshairManager)
#include "TestCrosshairManager.moc"
