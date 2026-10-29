#include <QtTest>

#include <QFileInfo>
#include <QSignalSpy>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vtkActor.h>
#include <vtkSmartPointer.h>

#include "ISceneService.h"
#include "SignalBus.h"
#include "ViewerService.h"

namespace {

/// The scene is not the subject here: the test only needs to observe what the
/// viewer service asks it to display. The pointer is recorded without VTK
/// rendering, so the test stays headless.
class FakeSceneService : public ISceneService {
public:
    vtkRenderer* renderer() override { return nullptr; }

    void setTextureActor(vtkSmartPointer<vtkActor> actor) override {
        ++textureCalls;
        lastActor = actor;
    }
    void setPointCloudActor(vtkSmartPointer<vtkActor>) override {}
    void clear3DModel() override {}
    void clear2DTexture() override {}
    void clearPointCloud() override {}
    void resetToSingleRenderer() override {}
    void loadOBJwithMTL(const QString&, const QString&) override {}
    bool onLoadDicom(const QString&) override { return false; }

    vtkSmartPointer<vtkActor> loadImageActor(const QString&) override { return nullptr; }
    vtkSmartPointer<vtkActor> loadImageActor(const cv::Mat&) override { return nullptr; }
    void showImage(const QString&) override {}
    void showImage(const cv::Mat&) override {}

    bool isPointCloudVisible() const override { return false; }
    void setPointCloudVisible(bool) override {}

    void render() override { ++renderCalls; }

    void setPointCloudData(const std::vector<Point3D>&,
                           const std::vector<PointColor>& = {}) override {}
    void applyViewSettings(const QString&) override {}

    int textureCalls = 0;
    int renderCalls = 0;
    vtkSmartPointer<vtkActor> lastActor;
};

/// Writes a 4×4 PNG so the real Image2DLoader path (not a mock) is exercised.
bool writeFrame(const QString& path)
{
    const cv::Mat image(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    return cv::imwrite(path.toStdString(), image);
}

} // namespace

/**
 * @brief Contract of the 2-D paging service.
 *
 * The service owns the image list, the current index and the auto-navigation
 * timer; the viewer plugin only calls it. An off-by-one at either end of the
 * list silently shows the wrong slice, and a stale index after a new list is
 * how the viewer used to open the previous series' image.
 */
class TestViewerService : public QObject {
    Q_OBJECT

private slots:
    void defaultsAndFlags();
    void setImageListPublishesTheNewState();
    void loadResolvesTheImageAgainstTheCurrentDirectory();
    void loadWithoutAValidIndexDoesNothing();
    void pagingStopsAtBothEnds();
    void autoNavigationTogglesDirection();
    void autoTimeoutAdvancesAndStopsAtTheEnd();
    void autoTimeoutStopsAtTheStartWhenGoingBackwards();
};

void TestViewerService::defaultsAndFlags()
{
    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);

    QCOMPARE(viewer.getCurrent2DImagePath(), QString());
    QCOMPARE(viewer.getCurrentAIMode(), AIMode::None);
    QVERIFY(viewer.isUpdateSceneEnabled());

    viewer.setCurrent2DImagePath(QStringLiteral("C:/images/a.png"));
    QCOMPARE(viewer.getCurrent2DImagePath(), QStringLiteral("C:/images/a.png"));

    viewer.setAIMode(AIMode::Tracking);
    QCOMPARE(viewer.getCurrentAIMode(), AIMode::Tracking);

    viewer.setUpdateSceneEnabled(false);
    QVERIFY(!viewer.isUpdateSceneEnabled());
}

void TestViewerService::setImageListPublishesTheNewState()
{
    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);

    QSignalSpy listSpy(&bus, &SignalBus::imageListUpdated);
    QSignalSpy indexSpy(&bus, &SignalBus::imageIndexChanged);

    viewer.setImageList({QStringLiteral("a.png"), QStringLiteral("b.png")}, 1);

    QCOMPARE(listSpy.count(), 1);
    QCOMPARE(listSpy.first().at(0).toStringList(),
             (QStringList{QStringLiteral("a.png"), QStringLiteral("b.png")}));
    QCOMPARE(listSpy.first().at(1).toInt(), 1);

    QCOMPARE(indexSpy.count(), 1);
    QCOMPARE(indexSpy.first().at(0).toInt(), 1);
    QCOMPARE(indexSpy.first().at(1).toInt(), 2);
}

void TestViewerService::loadResolvesTheImageAgainstTheCurrentDirectory()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString first = dir.filePath(QStringLiteral("frame1.png"));
    QVERIFY(writeFrame(first));

    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);
    QSignalSpy indexSpy(&bus, &SignalBus::imageIndexChanged);

    // The current path only supplies the directory: the list holds file
    // names, which is what the previous behaviour got wrong.
    viewer.setCurrent2DImagePath(dir.filePath(QStringLiteral("placeholder.png")));
    viewer.setImageList({QStringLiteral("frame1.png"), QStringLiteral("frame2.png")}, 0);
    viewer.loadCurrentIndexImage();

    QCOMPARE(viewer.getCurrent2DImagePath(), first);
    QCOMPARE(scene.textureCalls, 1);
    QVERIFY(scene.lastActor != nullptr);
    QCOMPARE(scene.renderCalls, 1);

    // setImageList + loadCurrentIndexImage.
    QCOMPARE(indexSpy.count(), 2);
    QCOMPARE(indexSpy.last().at(0).toInt(), 0);
    QCOMPARE(indexSpy.last().at(1).toInt(), 2);
}

void TestViewerService::loadWithoutAValidIndexDoesNothing()
{
    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);

    viewer.setCurrent2DImagePath(QStringLiteral("C:/images/a.png"));

    viewer.setImageList({QStringLiteral("a.png")}, -1);
    viewer.loadCurrentIndexImage();
    QCOMPARE(scene.textureCalls, 0);
    QCOMPARE(scene.renderCalls, 0);

    viewer.setImageList({}, 0);
    viewer.loadCurrentIndexImage();
    QCOMPARE(scene.textureCalls, 0);
    QCOMPARE(scene.renderCalls, 0);
}

void TestViewerService::pagingStopsAtBothEnds()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString a = dir.filePath(QStringLiteral("a.png"));
    const QString b = dir.filePath(QStringLiteral("b.png"));
    const QString c = dir.filePath(QStringLiteral("c.png"));
    QVERIFY(writeFrame(a));
    QVERIFY(writeFrame(b));
    QVERIFY(writeFrame(c));

    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);
    viewer.setCurrent2DImagePath(dir.filePath(QStringLiteral("placeholder.png")));
    viewer.setImageList({QStringLiteral("a.png"), QStringLiteral("b.png"), QStringLiteral("c.png")}, 0);

    // Going back from the first image must not wrap to the last one.
    viewer.onPrevImage();
    QCOMPARE(viewer.getCurrent2DImagePath(), dir.filePath(QStringLiteral("placeholder.png")));
    QCOMPARE(scene.renderCalls, 0);

    viewer.onNextImage();
    QCOMPARE(viewer.getCurrent2DImagePath(), b);
    QCOMPARE(scene.renderCalls, 1);

    viewer.onNextImage();
    QCOMPARE(viewer.getCurrent2DImagePath(), c);
    QCOMPARE(scene.renderCalls, 2);

    // Going forward from the last image must stay put.
    viewer.onNextImage();
    QCOMPARE(viewer.getCurrent2DImagePath(), c);
    QCOMPARE(scene.renderCalls, 2);

    viewer.onPrevImage();
    QCOMPARE(viewer.getCurrent2DImagePath(), b);
    QCOMPARE(scene.renderCalls, 3);
}

void TestViewerService::autoNavigationTogglesDirection()
{
    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);
    QSignalSpy autoSpy(&bus, &SignalBus::autoNavigationChanged);

    viewer.onAutoNext();
    QCOMPARE(autoSpy.count(), 1);
    QCOMPARE(autoSpy.last().at(0).toBool(), true);
    QCOMPARE(autoSpy.last().at(1).toBool(), true);

    viewer.onAutoNext(); // same direction again → stop
    QCOMPARE(autoSpy.count(), 2);
    QCOMPARE(autoSpy.last().at(0).toBool(), false);
    QCOMPARE(autoSpy.last().at(1).toBool(), true);

    viewer.onAutoPrev();
    QCOMPARE(autoSpy.count(), 3);
    QCOMPARE(autoSpy.last().at(0).toBool(), true);
    QCOMPARE(autoSpy.last().at(1).toBool(), false);

    viewer.onAutoPrev(); // same direction again → stop
    QCOMPARE(autoSpy.count(), 4);
    QCOMPARE(autoSpy.last().at(0).toBool(), false);
    QCOMPARE(autoSpy.last().at(1).toBool(), false);

    // While a direction runs, the other button switches it instead of
    // stopping: that is what makes the two buttons feel like a toggle.
    viewer.onAutoNext();
    viewer.onAutoPrev();
    QCOMPARE(autoSpy.count(), 6);
    QCOMPARE(autoSpy.last().at(0).toBool(), true);
    QCOMPARE(autoSpy.last().at(1).toBool(), false);

    viewer.onAutoNext();
    QCOMPARE(autoSpy.count(), 7);
    QCOMPARE(autoSpy.last().at(0).toBool(), true);
    QCOMPARE(autoSpy.last().at(1).toBool(), true);
}

void TestViewerService::autoTimeoutAdvancesAndStopsAtTheEnd()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QVERIFY(writeFrame(dir.filePath(QStringLiteral("a.png"))));
    QVERIFY(writeFrame(dir.filePath(QStringLiteral("b.png"))));

    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);
    QSignalSpy autoSpy(&bus, &SignalBus::autoNavigationChanged);

    viewer.setCurrent2DImagePath(dir.filePath(QStringLiteral("placeholder.png")));
    viewer.setImageList({QStringLiteral("a.png"), QStringLiteral("b.png")}, 0);
    viewer.onAutoNext();
    QCOMPARE(autoSpy.count(), 1);

    QVERIFY(QMetaObject::invokeMethod(&viewer, "onAutoTimerTimeout", Qt::DirectConnection));
    QCOMPARE(viewer.getCurrent2DImagePath(), dir.filePath(QStringLiteral("b.png")));
    // Still active: the last image has not been reached yet.
    QCOMPARE(autoSpy.count(), 1);

    QVERIFY(QMetaObject::invokeMethod(&viewer, "onAutoTimerTimeout", Qt::DirectConnection));
    QCOMPARE(viewer.getCurrent2DImagePath(), dir.filePath(QStringLiteral("b.png")));
    QCOMPARE(autoSpy.count(), 2);
    QCOMPARE(autoSpy.last().at(0).toBool(), false);
    QCOMPARE(autoSpy.last().at(1).toBool(), true);
}

void TestViewerService::autoTimeoutStopsAtTheStartWhenGoingBackwards()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QVERIFY(writeFrame(dir.filePath(QStringLiteral("a.png"))));

    SignalBus bus;
    FakeSceneService scene;
    ViewerService viewer(&scene, &bus);
    QSignalSpy autoSpy(&bus, &SignalBus::autoNavigationChanged);

    viewer.setCurrent2DImagePath(dir.filePath(QStringLiteral("a.png")));
    viewer.setImageList({QStringLiteral("a.png")}, 0);
    viewer.onAutoPrev();
    QCOMPARE(autoSpy.count(), 1);

    QVERIFY(QMetaObject::invokeMethod(&viewer, "onAutoTimerTimeout", Qt::DirectConnection));

    QCOMPARE(autoSpy.count(), 2);
    QCOMPARE(autoSpy.last().at(0).toBool(), false);
    QCOMPARE(autoSpy.last().at(1).toBool(), false);
}

QTEST_GUILESS_MAIN(TestViewerService)
#include "TestViewerService.moc"
