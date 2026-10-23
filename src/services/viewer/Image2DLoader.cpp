#include "Image2DLoader.h"
#include <QFileInfo>
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkPNGReader.h>
#include <vtkJPEGReader.h>
#include <vtkBMPReader.h>
#include <vtkImageImport.h>
#include <opencv2/imgproc.hpp>

vtkSmartPointer<vtkActor> Image2DLoader::load(const QString& fileName) {
    if (fileName.isEmpty()) return nullptr;

    vtkNew<vtkImageReader2Factory> readerFactory;
    vtkSmartPointer<vtkImageReader2> reader = readerFactory->CreateImageReader2(fileName.toStdString().c_str());
    
    if (!reader) {
        // Fallback or manual check if factory fails
        if (fileName.endsWith(".png", Qt::CaseInsensitive)) reader = vtkSmartPointer<vtkPNGReader>::New();
        else if (fileName.endsWith(".bmp", Qt::CaseInsensitive)) reader = vtkSmartPointer<vtkBMPReader>::New();
        else reader = vtkSmartPointer<vtkJPEGReader>::New();
    }

    reader->SetFileName(fileName.toStdString().c_str());
    reader->Update();

    vtkImageData *imageData = reader->GetOutput();
    if (!imageData || imageData->GetNumberOfPoints() == 0) return nullptr;

    int *dims = imageData->GetDimensions();
    int width = dims[0];
    int height = dims[1];
    double aspect = (double)width / (double)height;

    double planeWidth = aspect;
    double planeHeight = 1.0;
    
    vtkNew<vtkPlaneSource> plane;
    plane->SetOrigin(-planeWidth / 2.0, -planeHeight / 2.0, 0.0);
    plane->SetPoint1(planeWidth / 2.0, -planeHeight / 2.0, 0.0);
    plane->SetPoint2(-planeWidth / 2.0, planeHeight / 2.0, 0.0);
    plane->SetResolution(1, 1);
    plane->Update();

    vtkNew<vtkTexture> texture;
    texture->SetInputConnection(reader->GetOutputPort());
    texture->InterpolateOn();

    vtkNew<vtkPolyDataMapper> planeMapper;
    planeMapper->SetInputConnection(plane->GetOutputPort());

    vtkSmartPointer<vtkActor> texturePlaneActor = vtkSmartPointer<vtkActor>::New();
    texturePlaneActor->SetMapper(planeMapper);
    texturePlaneActor->SetTexture(texture);
    texturePlaneActor->GetProperty()->SetLighting(false);

    return texturePlaneActor;
}

vtkSmartPointer<vtkActor> Image2DLoader::loadFromMat(const cv::Mat& image) {
    if (image.empty()) return nullptr;

    cv::Mat processedImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, processedImage, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, processedImage, cv::COLOR_BGRA2RGBA);
    } else {
        processedImage = image.clone();
    }
    
    // VTK's origin is bottom-left, OpenCV is top-left
    cv::flip(processedImage, processedImage, 0);

    vtkNew<vtkImageImport> importer;
    importer->SetDataSpacing(1, 1, 1);
    importer->SetDataOrigin(0, 0, 0);
    importer->SetWholeExtent(0, processedImage.cols - 1, 0, processedImage.rows - 1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(processedImage.channels());
    importer->CopyImportVoidPointer(processedImage.data, processedImage.total() * processedImage.elemSize());
    importer->Update();

    vtkImageData *imageData = importer->GetOutput();
    if (!imageData || imageData->GetNumberOfPoints() == 0) return nullptr;

    int *dims = imageData->GetDimensions();
    int width = dims[0];
    int height = dims[1];
    double aspect = (double)width / (double)height;

    double planeWidth = aspect;
    double planeHeight = 1.0;
    
    vtkNew<vtkPlaneSource> plane;
    plane->SetOrigin(-planeWidth / 2.0, -planeHeight / 2.0, 0.0);
    plane->SetPoint1(planeWidth / 2.0, -planeHeight / 2.0, 0.0);
    plane->SetPoint2(-planeWidth / 2.0, planeHeight / 2.0, 0.0);
    plane->SetResolution(1, 1);
    plane->Update();

    vtkNew<vtkTexture> texture;
    texture->SetInputConnection(importer->GetOutputPort());
    texture->InterpolateOn();

    vtkNew<vtkPolyDataMapper> planeMapper;
    planeMapper->SetInputConnection(plane->GetOutputPort());

    vtkSmartPointer<vtkActor> texturePlaneActor = vtkSmartPointer<vtkActor>::New();
    texturePlaneActor->SetMapper(planeMapper);
    texturePlaneActor->SetTexture(texture);
    texturePlaneActor->GetProperty()->SetLighting(false);

    return texturePlaneActor;
}
