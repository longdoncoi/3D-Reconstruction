#include <QtTest>

#include <QHash>
#include <QString>
#include <vector>

#include <opencv2/core.hpp>

#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkRenderer.h>

#include "IUserPreferences.h"
#include "PointCloudData.h"
#include "SceneService.h"

namespace {

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
    void modelLifecycle();
    void viewSettings();
    void resetToSingleRenderer();
};

void TestSceneService::creationAndDefaults() {
    SceneService scene(nullptr, nullptr);
    QVERIFY(scene.renderer() != nullptr);
    QVERIFY(scene.vtkWidget() == nullptr);
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

void TestSceneService::modelLifecycle() {
    SceneService scene(nullptr, nullptr);
    scene.clear3DModel();
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
