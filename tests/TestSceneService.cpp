#include <QtTest>

#include <QDir>
#include <QFile>
#include <QHash>
#include <QString>
#include <QTemporaryDir>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include "IUserPreferences.h"
#include "PointCloudData.h"
#include "SceneService.h"

namespace {

/// A render window whose Render() is a no-op: the scene logic (layout,
/// actors, camera, interactor style) runs exactly as in production, but the
/// test does not need an OpenGL surface or a visible native window.
class HeadlessRenderWindow : public vtkRenderWindow {
public:
    static HeadlessRenderWindow* New();
    vtkTypeMacro(HeadlessRenderWindow, vtkRenderWindow);

    void Render() override { ++renderCalls; }

    int renderCalls = 0;
};
vtkStandardNewMacro(HeadlessRenderWindow);

bool writeText(const QString &path, const char *text)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return false;
    return file.write(text) >= 0;
}

bool writeTriangleObj(const QString &path)
{
    return writeText(path,
                     "v 0.0 0.0 0.0\n"
                     "v 1.0 0.0 0.0\n"
                     "v 0.0 1.0 0.0\n"
                     "f 1 2 3\n");
}

bool writeFrame(const QString &path)
{
    const cv::Mat image(8, 8, CV_8UC3, cv::Scalar(30, 20, 10));
    return cv::imwrite(path.toStdString(), image);
}

class FakeUserPreferences : public IUserPreferences {
public:
    QHash<QString, QString> prefs;

    QString getUserPref(const QString &username, const QString &key, const QString &defaultValue) const override {
        return prefs.value(username + "/" + key, defaultValue);
    }
    void setUserPref(const QString &username, const QString &key, const QString &value) override {
        prefs.insert(username + "/" + key, value);
    }
    QString getUserSecret(const QString &username, const QString &key, const QString &defaultValue = {}) const override {
        return prefs.value("sec/" + username + "/" + key, defaultValue);
    }
    void setUserSecret(const QString &username, const QString &key, const QString &value) override {
        prefs.insert("sec/" + username + "/" + key, value);
    }
};

} // namespace

class TestSceneService : public QObject {
    Q_OBJECT

private slots:
    void creationAndDefaults();
    void pointCloudLifecycle();
    void pointCloudWithColors();
    void textureAndImageActors();
    void imagePathOverload();
    void modelLifecycle();
    void modelLoadAndClear();
    void viewSettings();
    void dicomRejectsEmptyAndMissingPaths();
    void dicomSeriesBuildsTheMprLayout();
    void explicitRenderWindowAndPanStyle();
    void resetToSingleRenderer();
};

void TestSceneService::creationAndDefaults() {
    SceneService scene(nullptr, nullptr);
    QVERIFY(scene.renderer() != nullptr);
    QVERIFY(!scene.isPointCloudVisible());
    // Safe to call render() in headless mode
    scene.render();
}

void TestSceneService::pointCloudLifecycle() {
    SceneService scene(nullptr, nullptr);
    std::vector<Point3D> pts = { {0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f} };
    scene.setPointCloudData(pts);
    QVERIFY(scene.isPointCloudVisible());
    QVERIFY(scene.renderer()->GetActors()->GetNumberOfItems() > 0);

    scene.setPointCloudVisible(false);
    QVERIFY(!scene.isPointCloudVisible());

    scene.clearPointCloud();
    QVERIFY(!scene.isPointCloudVisible());
}

void TestSceneService::pointCloudWithColors() {
    SceneService scene(nullptr, nullptr);
    std::vector<Point3D> pts = { {0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f} };
    std::vector<PointColor> colors = { {255, 0, 0}, {0, 255, 0} };
    scene.setPointCloudData(pts, colors);
    QVERIFY(scene.isPointCloudVisible());

    // Empty point cloud clears the actors
    scene.setPointCloudData({});
    QVERIFY(!scene.isPointCloudVisible());
}

void TestSceneService::textureAndImageActors() {
    SceneService scene(nullptr, nullptr);
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));

    auto actor = scene.loadImageActor(mat);
    QVERIFY(actor != nullptr);

    scene.setTextureActor(actor);
    scene.showImage(mat);
    scene.clear2DTexture();
}

void TestSceneService::imagePathOverload() {
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString frame = dir.filePath(QStringLiteral("frame.png"));
    QVERIFY(writeFrame(frame));

    SceneService scene(nullptr, nullptr);

    // A missing file is not an error: the loader reports it as "no actor",
    // and the scene must not end up with a textured plane for it.
    QVERIFY(scene.loadImageActor(dir.filePath(QStringLiteral("missing.png"))) == nullptr);

    QVERIFY(scene.loadImageActor(frame) != nullptr);
    scene.showImage(frame);
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 1);

    // Showing a cv::Mat replaces the previous texture instead of stacking a
    // second plane on top of it.
    scene.showImage(cv::Mat(16, 16, CV_8UC3, cv::Scalar(1, 2, 3)));
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 1);

    scene.clear2DTexture();
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 0);

    // Clearing twice (and clearing a never-set texture) is safe.
    scene.clear2DTexture();
    scene.setTextureActor(nullptr);
}

void TestSceneService::modelLifecycle() {
    SceneService scene(nullptr, nullptr);
    scene.clear3DModel();
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 0);
}

void TestSceneService::modelLoadAndClear() {
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString obj = dir.filePath(QStringLiteral("triangle.obj"));
    const QString mtl = dir.filePath(QStringLiteral("triangle.mtl"));
    QVERIFY(writeTriangleObj(obj));
    QVERIFY(writeText(mtl, "newmtl material0\n"
                           "Ka 1.000000 1.000000 1.000000\n"
                           "Kd 0.700000 0.700000 0.700000\n"
                           "Ks 0.000000 0.000000 0.000000\n"
                           "d 1.000000\n"
                           "illum 2\n"));

    SceneService scene(nullptr, nullptr);
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 0);

    scene.loadOBJwithMTL(obj, mtl);
    const int loadedActors = scene.renderer()->GetActors()->GetNumberOfItems();
    QVERIFY(loadedActors > 0);

    // A second load replaces the first model instead of accumulating actors.
    scene.loadOBJwithMTL(obj, mtl);
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), loadedActors);

    scene.clear3DModel();
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 0);

    // A missing path must leave the previous state untouched.
    scene.loadOBJwithMTL(dir.filePath(QStringLiteral("missing.obj")), QString());
    QCOMPARE(scene.renderer()->GetActors()->GetNumberOfItems(), 0);
}

void TestSceneService::viewSettings() {
    // Null prefs is safe
    SceneService scene1(nullptr, nullptr);
    scene1.applyViewSettings(QStringLiteral("test_user"));

    FakeUserPreferences prefs;
    SceneService scene2(&prefs, nullptr);

    // Dark background (0)
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_bg_color"), QStringLiteral("0"));
    scene2.applyViewSettings(QStringLiteral("alice"));

    // Light background (1)
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_bg_color"), QStringLiteral("1"));
    scene2.applyViewSettings(QStringLiteral("alice"));

    // Gray background (2)
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_bg_color"), QStringLiteral("2"));
    scene2.applyViewSettings(QStringLiteral("alice"));

    // Grid on and off
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_show_grid"), QStringLiteral("true"));
    scene2.applyViewSettings(QStringLiteral("alice"));

    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_show_grid"), QStringLiteral("false"));
    scene2.applyViewSettings(QStringLiteral("alice"));
    QCOMPARE(scene2.renderer()->GetActors()->GetNumberOfItems(), 0);
}

void TestSceneService::dicomRejectsEmptyAndMissingPaths() {
    SceneService scene(nullptr, nullptr);

    // A cancelled file dialog is not an error and must not touch the scene.
    QVERIFY(!scene.onLoadDicom(QString()));

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QVERIFY(!scene.onLoadDicom(dir.path())); // empty directory
    QVERIFY(!scene.onLoadDicom(dir.filePath(QStringLiteral("does-not-exist"))));
}

void TestSceneService::dicomSeriesBuildsTheMprLayout() {
    // Three slices from the repository's sample series: enough for a volume
    // and for the MPR layout, without copying the 64 MB series into a temp
    // directory (or writing DicomLoader's cache into the source tree).
    const QDir source(QStringLiteral(TEST_DICOM_DIR));
    QVERIFY2(source.exists(), TEST_DICOM_DIR);

    const QStringList slices =
        source.entryList({QStringLiteral("MR.*")}, QDir::Files, QDir::Name);
    QVERIFY(slices.size() >= 3);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    for (int i = 0; i < 3; ++i) {
        QVERIFY2(QFile::copy(source.absoluteFilePath(slices.at(i)),
                             dir.filePath(slices.at(i))),
                 qPrintable(slices.at(i)));
    }

    vtkNew<HeadlessRenderWindow> window;
    vtkNew<vtkRenderWindowInteractor> interactor;
    window->SetInteractor(interactor);

    FakeUserPreferences prefs;
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_show_axes"), QStringLiteral("true"));

    SceneService scene(&prefs, nullptr, nullptr, window);
    QCOMPARE(window->GetRenderers()->GetNumberOfItems(), 1); // the 3-D renderer

    // Create the orientation marker before the DICOM load so the DICOM branch
    // that repositions it (lower-left of the 3-D quadrant) is exercised.
    // The marker adds a renderer of its own to the window.
    scene.applyViewSettings(QStringLiteral("alice"));
    const int renderersBeforeLoad = window->GetRenderers()->GetNumberOfItems();

    QVERIFY(scene.onLoadDicom(dir.path()));
    QCOMPARE(window->GetRenderers()->GetNumberOfItems(), renderersBeforeLoad + 3);
    QVERIFY(window->renderCalls > 0);

    // Back to a single renderer: the three MPR renderers must be removed from
    // the window and the 3-D view must cover the whole viewport again.
    scene.resetToSingleRenderer();
    QCOMPARE(window->GetRenderers()->GetNumberOfItems(), renderersBeforeLoad);
    double vp[4] = {0, 0, 0, 0};
    scene.renderer()->GetViewport(vp);
    QCOMPARE(vp[2], 1.0);
    QCOMPARE(vp[3], 1.0);

    // A service without a render window still owns the volume and the
    // crosshair manager: the load must take the "no window" branches, and the
    // destructor must clean the manager up. The cache written by the load
    // above makes this second read cheap.
    {
        SceneService headless(nullptr, nullptr);
        QVERIFY(headless.onLoadDicom(dir.path()));
        // A second series replaces the existing crosshair manager instead of
        // leaking it; the cached volume makes this read cheap.
        QVERIFY(headless.onLoadDicom(dir.path()));
    }
}

void TestSceneService::explicitRenderWindowAndPanStyle() {
    vtkNew<HeadlessRenderWindow> window;
    vtkNew<vtkRenderWindowInteractor> interactor;
    window->SetInteractor(interactor);

    FakeUserPreferences prefs;
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_show_axes"), QStringLiteral("true"));

    SceneService scene(&prefs, nullptr, nullptr, window);
    QCOMPARE(window->GetRenderers()->GetNumberOfItems(), 1);

    scene.render();
    QCOMPARE(window->renderCalls, 1);

    // resetToSingleRenderer installs the pan style on the window's interactor
    // and asks it to repaint.
    scene.resetToSingleRenderer();
    QVERIFY(interactor->GetInteractorStyle() != nullptr);
    QVERIFY(interactor->GetInteractorStyle()->IsA("PanStyle"));
    QCOMPARE(window->renderCalls, 2);

    // The orientation marker is created once, then re-enabled / disabled by
    // later settings, including the branch that only flips its state.
    scene.applyViewSettings(QStringLiteral("alice"));
    scene.applyViewSettings(QStringLiteral("alice"));
    prefs.setUserPref(QStringLiteral("alice"), QStringLiteral("view_show_axes"), QStringLiteral("false"));
    scene.applyViewSettings(QStringLiteral("alice"));

    // A point cloud renders through the injected window, too.
    scene.setPointCloudData({{0.0f, 0.0f, 0.0f}});
    QVERIFY(window->renderCalls > 2);
}

void TestSceneService::resetToSingleRenderer() {
    SceneService scene(nullptr, nullptr);
    scene.resetToSingleRenderer();

    double vp[4] = {0, 0, 0, 0};
    scene.renderer()->GetViewport(vp);
    QCOMPARE(vp[0], 0.0);
    QCOMPARE(vp[1], 0.0);
    QCOMPARE(vp[2], 1.0);
    QCOMPARE(vp[3], 1.0);
}

QTEST_MAIN(TestSceneService)
#include "TestSceneService.moc"
