#include <QtTest>

#include <cmath>

#include "CrosshairGeometry.h"

/**
 * @brief Unit tests for CrosshairGeometry (pure DICOM-slice math used by
 *        SceneService's crosshair reslicing). No VTK objects required.
 *
 * Contract under test:
 *  - lookup accessors return valid, non-null data for all 3 orientations,
 *  - normalized<->world round-trips are stable,
 *  - the axial orientation flips the Y axis (DICOM convention: superior
 *    is up), while sagittal/coronal map linearly.
 *
 * Note: worldToNormalized() adds a 1e-9 epsilon to denominators, so linear
 * expectations are compared with a small absolute tolerance.
 */
namespace {
constexpr double kTol = 1e-6;
bool closeTo(double a, double b) { return std::abs(a - b) < kTol; }
}

class TestCrosshairGeometry : public QObject {
    Q_OBJECT

private slots:
    void accessorsAreValidForAllOrientations();
    void axialRoundTripIsStable();
    void axialFlipsYAxis();
    void sagittalMapsYZLinearly();
};

void TestCrosshairGeometry::accessorsAreValidForAllOrientations()
{
    for (int ori = 0; ori < 3; ++ori) {
        QVERIFY(CrosshairGeometry::resliceAxes(ori) != nullptr);
        QVERIFY(CrosshairGeometry::lineColor(ori) != nullptr);
        const char *labels[4] = {};
        CrosshairGeometry::labels(ori, labels);
        for (int i = 0; i < 4; ++i)
            QVERIFY(labels[i] != nullptr);
    }
}

void TestCrosshairGeometry::axialRoundTripIsStable()
{
    const double bounds[6] = {-10.0, 10.0, -20.0, 20.0, -30.0, 30.0};
    const double world[3] = {3.0, -7.0, 12.0};

    double nx = 0, ny = 0;
    CrosshairGeometry::worldToNormalized(ORI_AXIAL, world, bounds, nx, ny);

    double back[3] = {0, 0, 0};
    // Axial: Z (slice position) is deliberately preserved from oldWorld,
    // not reconstructed from the normalized pair.
    const double oldWorld[3] = {0, 0, 12.0};
    CrosshairGeometry::normalizedToWorld(ORI_AXIAL, nx, ny, bounds, oldWorld, back);

    QVERIFY(closeTo(back[0], world[0]));
    QVERIFY(closeTo(back[1], world[1]));
    QVERIFY(closeTo(back[2], oldWorld[2]));
}

void TestCrosshairGeometry::axialFlipsYAxis()
{
    const double bounds[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    const double maxY[3] = {0.5, 1.0, 0.5}; // top edge in world Y

    double nx = 0, ny = 0;
    CrosshairGeometry::worldToNormalized(ORI_AXIAL, maxY, bounds, nx, ny);

    // World Y == maxY maps to normalized 0 (DICOM axial: superior up).
    QVERIFY(closeTo(ny, 0.0));
}

void TestCrosshairGeometry::sagittalMapsYZLinearly()
{
    const double bounds[6] = {0.0, 10.0, 0.0, 100.0, 0.0, 1000.0};
    const double world[3] = {5.0, 25.0, 750.0}; // X ignored for sagittal

    double nx = 0, ny = 0;
    CrosshairGeometry::worldToNormalized(ORI_SAGITTAL, world, bounds, nx, ny);
    QVERIFY(closeTo(nx, 0.25)); // Y: 25/100
    QVERIFY(closeTo(ny, 0.75)); // Z: 750/1000

    double back[3] = {0, 0, 0};
    const double oldWorld[3] = {42.0, 0, 0}; // sagittal keeps old X
    CrosshairGeometry::normalizedToWorld(ORI_SAGITTAL, nx, ny, bounds, oldWorld, back);
    QVERIFY(closeTo(back[0], 42.0));
    QVERIFY(closeTo(back[1], 25.0));
    QVERIFY(closeTo(back[2], 750.0));
}

QTEST_GUILESS_MAIN(TestCrosshairGeometry)
#include "TestCrosshairGeometry.moc"
