#include <QtTest>

#include <QDir>
#include <QFile>
#include <QImage>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Image2DLoader.h"
#include "Model3DLoader.h"

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkTexture.h>

/**
 * @brief Contract of the viewer loaders (Image2DLoader, Model3DLoader).
 *
 * Both turn a file into VTK actors for the scene.  They were the only
 * service translation units with no test at all, so a broken loader (a wrong
 * aspect ratio, a missing file, an unsupported format) would only show up as
 * an empty 3-D view in the application.
 */
class TestViewerLoaders : public QObject {
    Q_OBJECT

private slots:
    void imageLoaderRejectsEmptyAndMissingFiles();
    void imageLoaderBuildsATexturedPlane();
    void imageLoaderAcceptsOpenCvMatrices();
    void modelLoaderRejectsMissingAndUnsupportedFiles();
    void modelLoaderBuildsAnActorFromAnObjFile();
};

namespace {

/// A tiny valid OBJ file (one triangle).
bool writeTriangleObj(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    file.write("v 0.0 0.0 0.0\n"
               "v 1.0 0.0 0.0\n"
               "v 0.0 1.0 0.0\n"
               "f 1 2 3\n");
    return true;
}

} // namespace

void TestViewerLoaders::imageLoaderRejectsEmptyAndMissingFiles()
{
    QVERIFY(Image2DLoader::load(QString()) == nullptr);
    QVERIFY(Image2DLoader::load(QStringLiteral("no-such-file.png")) == nullptr);
}

void TestViewerLoaders::imageLoaderBuildsATexturedPlane()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString path = dir.filePath(QStringLiteral("sample.png"));
    QImage image(64, 32, QImage::Format_RGB888);
    image.fill(Qt::red);
    QVERIFY(image.save(path, "PNG"));

    const auto actor = Image2DLoader::load(path);
    QVERIFY(actor != nullptr);
    QVERIFY(actor->GetMapper() != nullptr);
    QVERIFY(actor->GetTexture() != nullptr);
    // Lighting is off: the 2-D image must be shown as-is.
    QCOMPARE(actor->GetProperty()->GetLighting(), 0);

    // The plane keeps the aspect ratio of the image (64x32 → 2.0).
    double bounds[6] = {0, 0, 0, 0, 0, 0};
    actor->GetBounds(bounds);
    const double width = bounds[1] - bounds[0];
    const double height = bounds[3] - bounds[2];
    QVERIFY2(qAbs(height - 1.0) < 1e-6, qPrintable(QStringLiteral("height=%1").arg(height)));
    QVERIFY2(qAbs(width - 2.0) < 1e-6, qPrintable(QStringLiteral("width=%1").arg(width)));
}

void TestViewerLoaders::imageLoaderAcceptsOpenCvMatrices()
{
    QVERIFY(Image2DLoader::loadFromMat(cv::Mat()) == nullptr);

    cv::Mat bgr(20, 40, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto actor = Image2DLoader::loadFromMat(bgr);
    QVERIFY(actor != nullptr);
    QVERIFY(actor->GetMapper() != nullptr);
    QVERIFY(actor->GetTexture() != nullptr);

    cv::Mat rgba(20, 40, CV_8UC4, cv::Scalar(10, 20, 30, 40));
    const auto actorRgba = Image2DLoader::loadFromMat(rgba);
    QVERIFY(actorRgba != nullptr);
}

void TestViewerLoaders::modelLoaderRejectsMissingAndUnsupportedFiles()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    // Missing file.
    QVERIFY(Model3DLoader::load(dir.filePath(QStringLiteral("missing.obj")),
                                QString()).empty());

    // Unsupported format.
    const QString bogus = dir.filePath(QStringLiteral("model.xyz"));
    QFile file(bogus);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.write("nonsense");
    file.close();
    QVERIFY(Model3DLoader::load(bogus, QString()).empty());
}

void TestViewerLoaders::modelLoaderBuildsAnActorFromAnObjFile()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString obj = dir.filePath(QStringLiteral("triangle.obj"));
    QVERIFY(writeTriangleObj(obj));

    // Without a material file the loader falls back to the plain OBJ reader.
    const auto actors = Model3DLoader::load(obj, QString());
    QCOMPARE(static_cast<int>(actors.size()), 1);
    QVERIFY(actors.front() != nullptr);
    QVERIFY(actors.front()->GetMapper() != nullptr);
    QVERIFY(actors.front()->GetProperty() != nullptr);
}

QTEST_GUILESS_MAIN(TestViewerLoaders)
#include "TestViewerLoaders.moc"
