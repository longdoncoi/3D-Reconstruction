#include <QtTest>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Image2DLoader.h"
#include "Model3DLoader.h"
#include "DicomLoader.h"

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkTexture.h>

/**
 * @brief Contract of the viewer loaders (Image2DLoader, Model3DLoader, DicomLoader).
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
    void imageLoaderFallsBackToAnExplicitReader();
    void modelLoaderRejectsMissingAndUnsupportedFiles();
    void modelLoaderBuildsAnActorFromAnObjFile();
    void modelLoaderReadsEverySupportedFormat();
    void modelLoaderImportsAnObjWithItsMaterial();
    void modelLoaderRejectsAnEmptyMesh();
    void dicomLoaderHandlesInvalidAndEmptySeries();
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

/// Writes `text` to `path`; false when the file cannot be opened.
bool writeText(const QString &path, const char *text)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    return file.write(text) >= 0;
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

    // A single-channel frame (a mask, a DICOM slice) has no colour order to
    // convert: it must be cloned as-is, not pushed through a colour cvtColor
    // that assumes three channels.
    cv::Mat gray(20, 40, CV_8UC1, cv::Scalar(128));
    const auto actorGray = Image2DLoader::loadFromMat(gray);
    QVERIFY(actorGray != nullptr);
    QVERIFY(actorGray->GetMapper() != nullptr);
    QVERIFY(actorGray->GetTexture() != nullptr);
}

void TestViewerLoaders::imageLoaderFallsBackToAnExplicitReader()
{
    // vtkImageReader2Factory owns a table of extensions; it returns a null
    // reader for anything outside it. The loader must then pick a reader
    // itself — a null dereference here is a crash, not a viewer glitch. An
    // unknown extension is treated as JPEG, PNG/BMP are matched first (and
    // case-insensitively, like the factory itself).
    QVERIFY(Image2DLoader::createFallbackReader(QStringLiteral("scan.PNG"))
                ->IsA("vtkPNGReader"));
    QVERIFY(Image2DLoader::createFallbackReader(QStringLiteral("scan.Bmp"))
                ->IsA("vtkBMPReader"));
    QVERIFY(Image2DLoader::createFallbackReader(QStringLiteral("scan.weird"))
                ->IsA("vtkJPEGReader"));
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

void TestViewerLoaders::modelLoaderReadsEverySupportedFormat()
{
    // Every supported suffix has its own reader; a missing branch here is a
    // model that silently refuses to open in the viewer. The fixtures are the
    // smallest valid file of each format (one triangle).
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString stl = dir.filePath(QStringLiteral("triangle.stl"));
    QVERIFY(writeText(stl, "solid t\n"
                           "  facet normal 0 0 1\n"
                           "    outer loop\n"
                           "      vertex 0 0 0\n"
                           "      vertex 1 0 0\n"
                           "      vertex 0 1 0\n"
                           "    endloop\n"
                           "  endfacet\n"
                           "endsolid t\n"));

    const QString ply = dir.filePath(QStringLiteral("triangle.ply"));
    QVERIFY(writeText(ply, "ply\n"
                           "format ascii 1.0\n"
                           "element vertex 3\n"
                           "property float x\n"
                           "property float y\n"
                           "property float z\n"
                           "element face 1\n"
                           "property list uchar int vertex_indices\n"
                           "end_header\n"
                           "0 0 0\n"
                           "1 0 0\n"
                           "0 1 0\n"
                           "3 0 1 2\n"));

    const QString vtk = dir.filePath(QStringLiteral("triangle.vtk"));
    QVERIFY(writeText(vtk, "# vtk DataFile Version 3.0\n"
                           "triangle\n"
                           "ASCII\n"
                           "DATASET POLYDATA\n"
                           "POINTS 3 float\n"
                           "0 0 0\n"
                           "1 0 0\n"
                           "0 1 0\n"
                           "POLYGONS 1 4\n"
                           "3 0 1 2\n"));

    const QString vtp = dir.filePath(QStringLiteral("triangle.vtp"));
    QVERIFY(writeText(vtp, "<?xml version=\"1.0\"?>\n"
                           "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
                           "  <PolyData>\n"
                           "    <Piece NumberOfPoints=\"3\" NumberOfPolys=\"1\">\n"
                           "      <Points>\n"
                           "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n"
                           "          0 0 0 1 0 0 0 1 0\n"
                           "        </DataArray>\n"
                           "      </Points>\n"
                           "      <Polys>\n"
                           "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">0 1 2</DataArray>\n"
                           "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">3</DataArray>\n"
                           "      </Polys>\n"
                           "    </Piece>\n"
                           "  </PolyData>\n"
                           "</VTKFile>\n"));

    for (const QString &path : {stl, ply, vtk, vtp}) {
        const auto actors = Model3DLoader::load(path, QString());
        QVERIFY2(static_cast<int>(actors.size()) == 1,
                 qPrintable(QStringLiteral("no actor for %1").arg(path)));
        QVERIFY(actors.front()->GetMapper() != nullptr);
    }
}

void TestViewerLoaders::modelLoaderImportsAnObjWithItsMaterial()
{
    // With a material file the loader prefers vtkOBJImporter (it carries the
    // MTL's colours), and only falls back to the plain reader when the
    // importer produced no renderer. Both branches must yield the mesh.
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

    const auto actors = Model3DLoader::load(obj, mtl);
    QVERIFY(!actors.empty());
    QVERIFY(actors.front() != nullptr);
    QVERIFY(actors.front()->GetProperty() != nullptr);
}

void TestViewerLoaders::modelLoaderRejectsAnEmptyMesh()
{
    // A readable file with no geometry is not an error in the reader but is
    // an empty view: the loader must report "no mesh data" instead of adding
    // an actor with nothing in it.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString empty = dir.filePath(QStringLiteral("empty.stl"));
    QVERIFY(writeText(empty, "solid empty\nendsolid empty\n"));

    QVERIFY(Model3DLoader::load(empty, QString()).empty());
}

void TestViewerLoaders::dicomLoaderHandlesInvalidAndEmptySeries()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    // Empty directory returns nullptr
    auto volume = DicomLoader::loadSeries(dir.path());
    QVERIFY(!volume);
    // Non-existent directory returns nullptr
    auto missing = DicomLoader::loadSeries(dir.filePath(QStringLiteral("nonexistent")));
    QVERIFY(!missing);

    // Valid dummy image data exercises createSlice and createVolume
    vtkNew<vtkImageData> dummyData;
    dummyData->SetDimensions(4, 4, 4);
    dummyData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    double range[2] = {0.0, 255.0};

    auto slice = DicomLoader::createSlice(dummyData, 0, 100.0, 50.0);
    QVERIFY(slice != nullptr);
    auto vol = DicomLoader::createVolume(dummyData, range);
    QVERIFY(vol != nullptr);
}

QTEST_GUILESS_MAIN(TestViewerLoaders)
#include "TestViewerLoaders.moc"
