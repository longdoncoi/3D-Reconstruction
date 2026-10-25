#ifndef IMAGE2DLOADER_H
#define IMAGE2DLOADER_H

#include <QString>
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkImageData.h>
#include <vtkImageReader2.h>
#include <vtkPNGReader.h>
#include <vtkJPEGReader.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

#include "Global.h"
#include <opencv2/core.hpp>

class APP_EXPORT Image2DLoader {
public:
    static vtkSmartPointer<vtkActor> load(const QString& fileName);
    static vtkSmartPointer<vtkActor> loadFromMat(const cv::Mat& image);

    /// The reader the loader picks when vtkImageReader2Factory does not know
    /// the extension (it returns null for anything outside its table).
    /// Exposed so the decision can be tested without a file the factory
    /// refuses: an unknown extension is treated as JPEG rather than crashing
    /// on a null reader.
    static vtkSmartPointer<vtkImageReader2> createFallbackReader(const QString& fileName);
};

#endif // IMAGE2DLOADER_H
