// dicomloader.cpp
// Chiến lược: đọc từng file .dcm bằng vtkDICOMImageReader (single file mode),
// lấy InstanceNumber từ header để sort đúng thứ tự, rồi ghép thành volume
// bằng vtkImageAppend. Không phụ thuộc vào tên file hay directory-mode reader.

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2);

#include "DicomLoader.h"

#include <QDir>
#include <QCollator>
#include <QDebug>
#include <QString>
#include <QFile>

#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>

#include <vtkDICOMImageReader.h>
#include <vtkImageData.h>
#include <vtkImageAppend.h>
#include <vtkImageMapper3D.h>
#include <vtkImageActor.h>
#include <vtkImageProperty.h>
#include <vtkImageReslice.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkFixedPointVolumeRayCastMapper.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkImageCast.h>

#include <algorithm>
#include <vector>
#include <string>

// ---------------------------------------------------------------
// loadSeries: đọc từng slice, sort theo InstanceNumber, ghép volume
// ---------------------------------------------------------------
vtkSmartPointer<vtkImageData> DicomLoader::loadSeries(const QString &dirPath)
{
    QString cachePath = QDir(dirPath).absoluteFilePath("dicom_cache.vti");
    if (QFile::exists(cachePath)) {
        qDebug() << "DicomLoader: Found cache at" << cachePath;
        auto cacheReader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        cacheReader->SetFileName(cachePath.toLocal8Bit().constData());
        cacheReader->Update();
        if (cacheReader->GetOutput() && cacheReader->GetOutput()->GetNumberOfPoints() > 0) {
            return cacheReader->GetOutput();
        }
    }

    QDir dir(dirPath);
    QStringList filters; filters << "*.dcm" << "*.DCM" << "*.dicom" << "*.IMA";
    QStringList fileNames = dir.entryList(filters, QDir::Files, QDir::NoSort);

    if (fileNames.isEmpty()) {
        // Thử không có extension (một số DICOM không có .dcm)
        fileNames = dir.entryList(QDir::Files, QDir::NoSort);
    }

    if (fileNames.isEmpty()) {
        qDebug() << "DicomLoader: No files found in" << dirPath;
        return nullptr;
    }

    qDebug() << "DicomLoader: Found" << fileNames.size() << "files, reading metadata...";

    // --- Bước 1: đọc từng file, lấy InstanceNumber để sort ---
    struct SliceInfo {
        QString   path;
        int       instanceNumber;
        double    sliceLocation;   // backup sort key
        vtkSmartPointer<vtkImageData> imageData;
    };

    std::vector<SliceInfo> slices;
    slices.reserve(fileNames.size());

    int successCount = 0;
    for (const QString &fn : fileNames) {
        QString fullPath = dir.absoluteFilePath(fn);

        auto reader = vtkSmartPointer<vtkDICOMImageReader>::New();
        reader->SetFileName(fullPath.toLocal8Bit().constData());
        reader->Update();

        if (!reader->GetOutput() ||
            reader->GetOutput()->GetNumberOfPoints() < 1) {
            continue; // bỏ qua file lỗi
        }

        SliceInfo si;
        si.path           = fullPath;
        si.instanceNumber = 0; // Backup
        si.sliceLocation  = 0.0;

        // Lấy vị trí thực tế trong không gian 3D (Image Position Patient)
        float* pos = reader->GetImagePositionPatient();
        if (pos) {
            si.sliceLocation = pos[2]; // Tọa độ Z là chuẩn nhất để sort
        }

        // Deep copy imageData để reader có thể bị hủy
        si.imageData = vtkSmartPointer<vtkImageData>::New();
        si.imageData->DeepCopy(reader->GetOutput());

        slices.push_back(std::move(si));
        successCount++;
    }

    if (slices.empty()) {
        qDebug() << "DicomLoader: Failed to read any valid slice!";
        return nullptr;
    }

    qDebug() << "DicomLoader: Successfully read" << successCount << "slices";

    // --- Bước 2: Sort theo vị trí không gian (Z-origin) ---
    std::stable_sort(slices.begin(), slices.end(),
                     [](const SliceInfo& a, const SliceInfo& b){
                         return a.sliceLocation < b.sliceLocation;
                     });
    qDebug() << "DicomLoader: Sorted by Z-position (ImagePositionPatient)";

    // --- Bước 3: Lọc các slice có cùng kích thước (dimension) và ghép thành volume ---
    if (slices.empty()) return nullptr;

    // Lấy kích thước của slice đầu tiên làm chuẩn
    int baseDim[3];
    slices[0].imageData->GetDimensions(baseDim);

    vtkNew<vtkImageAppend> appender;
    appender->SetAppendAxis(2); // ghép theo trục Z

    int validSlices = 0;
    for (auto& si : slices) {
        int dim[3];
        si.imageData->GetDimensions(dim);
        if (dim[0] == baseDim[0] && dim[1] == baseDim[1]) {
            appender->AddInputData(si.imageData);
            validSlices++;
        } else {
            qDebug() << "DicomLoader: Bỏ qua slice do khác kích thước" << si.path << "(" << dim[0] << "x" << dim[1] << ")";
        }
    }

    if (validSlices == 0) return nullptr;

    appender->Update();

    if (!appender->GetOutput() ||
        appender->GetOutput()->GetNumberOfPoints() < 1) {
        qDebug() << "DicomLoader: ImageAppend failed!";
        return nullptr;
    }

    // Lấy spacing thực tế từ slice đầu tiên để set cho volume
    double spacing[3];
    slices[0].imageData->GetSpacing(spacing);
    // Spacing Z = khoảng cách giữa 2 slice liên tiếp
    if (slices.size() >= 2) {
        double o1[3], o2[3];
        slices[0].imageData->GetOrigin(o1);
        slices[1].imageData->GetOrigin(o2);
        double dz = std::abs(o2[2] - o1[2]);
        if (dz > 0.001) spacing[2] = dz;
        else            spacing[2] = spacing[0]; // fallback: isotropic
    }

    vtkSmartPointer<vtkImageData> volume = appender->GetOutput();
    volume->SetSpacing(spacing);

    // Debug final volume
    int* ext = volume->GetExtent();
    double range[2]; volume->GetScalarRange(range);
    qDebug() << "DicomLoader Volume:"
             << (ext[1]-ext[0]+1) << "x"
             << (ext[3]-ext[2]+1) << "x"
             << (ext[5]-ext[4]+1)
             << "| spacing:" << spacing[0] << spacing[1] << spacing[2]
             << "| range:" << range[0] << range[1];

    // Save to cache
    qDebug() << "DicomLoader: Saving cache to" << cachePath;
    auto writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(cachePath.toLocal8Bit().constData());
    writer->SetInputData(volume);
    writer->Write();

    return volume;
}

// ---------------------------------------------------------------
// createSlice: cắt lát MPR (nhận vtkImageData trực tiếp)
// orientation: 0=Sagittal, 1=Coronal, 2=Axial
// ---------------------------------------------------------------
vtkSmartPointer<vtkImageActor> DicomLoader::createSlice(
    vtkImageData* data, int orientation, double window, double level)
{
    double center[3];
    data->GetCenter(center);

    // Ma trận reslice chuẩn y khoa (RAS)
    static const double kAxes[3][16] = {
        // Sagittal (0): Anterior -> Right (+Y), Superior -> Top (+Z)
        { 0, 0, 1, 0,
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 0, 1 },
        // Coronal (1): Patient Left -> Right (+X), Superior -> Top (+Z)
        { 1, 0, 0, 0,
          0, 0, 1, 0,
          0, 1, 0, 0,
          0, 0, 0, 1 },
        // Axial (2): Patient Left -> Right (+X), Anterior -> Top (-Y)
        { 1, 0, 0, 0,
          0,-1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1 }
    };

    vtkNew<vtkMatrix4x4> axes;
    axes->DeepCopy(kAxes[orientation]);
    axes->SetElement(0, 3, center[0]);
    axes->SetElement(1, 3, center[1]);
    axes->SetElement(2, 3, center[2]);

    vtkNew<vtkImageReslice> reslice;
    reslice->SetInputData(data);           // ← SetInputData thay vì SetInputConnection
    reslice->SetOutputDimensionality(2);
    reslice->SetResliceAxes(axes);
    reslice->SetInterpolationModeToLinear();
    reslice->Update();

    auto actor = vtkSmartPointer<vtkImageActor>::New();
    actor->GetMapper()->SetInputConnection(reslice->GetOutputPort());
    actor->GetProperty()->SetColorWindow(window);
    actor->GetProperty()->SetColorLevel(level);
    return actor;
}

// ---------------------------------------------------------------
// createVolume: Volume rendering 3D
// ---------------------------------------------------------------
vtkSmartPointer<vtkVolume> DicomLoader::createVolume(
    vtkImageData* data, double range[2])
{
    double span = range[1] - range[0];

    vtkNew<vtkFixedPointVolumeRayCastMapper> mapper;
    mapper->SetInputData(data);
    mapper->SetAutoAdjustSampleDistances(0.5);
    mapper->SetInteractiveSampleDistance(0.5); // Khi xoay: nhanh, hơi nhòe chút
    mapper->SetSampleDistance(0.5);            // Khi dừng: rất nét
    mapper->SetBlendModeToMaximumIntensity(); 

    // Màu: đen → nâu (mô mềm) → kem (não) → trắng
    vtkNew<vtkColorTransferFunction> color;
    color->AddRGBPoint(range[0],              0.00, 0.00, 0.00);
    color->AddRGBPoint(range[0]+span*0.20,    0.40, 0.20, 0.10);
    color->AddRGBPoint(range[0]+span*0.50,    0.80, 0.75, 0.55);
    color->AddRGBPoint(range[1],              1.00, 1.00, 1.00);

    // Opacity: background trong suốt, cấu trúc não hiện dần
    vtkNew<vtkPiecewiseFunction> opacity;
    opacity->AddPoint(range[0],              0.00);
    opacity->AddPoint(range[0]+span*0.10,    0.00);
    opacity->AddPoint(range[0]+span*0.20,    0.08);
    opacity->AddPoint(range[0]+span*0.50,    0.25);
    opacity->AddPoint(range[1],              0.60);

    vtkNew<vtkVolumeProperty> prop;
    prop->SetColor(color);
    prop->SetScalarOpacity(opacity);
    prop->ShadeOn();
    prop->SetInterpolationTypeToLinear();
    prop->SetAmbient(0.3);
    prop->SetDiffuse(0.7);
    prop->SetSpecular(0.2);

    auto vol = vtkSmartPointer<vtkVolume>::New();
    vol->SetMapper(mapper);
    vol->SetProperty(prop);
    return vol;
}
