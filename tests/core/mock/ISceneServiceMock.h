#ifndef ISCENESERVICEMOCK_H
#define ISCENESERVICEMOCK_H

#include "include/ISceneService.h"
#include <vtkSmartPointer.h>
#include <vtkActor.h>

/**
 * @brief Mock implementation of ISceneService for architecture boundary tests.
 */
class MockSceneService : public ISceneService {
public:
    MockSceneService() = default;
    vtkSmartPointer<vtkActor> loadImageActor(const QString&) override { return nullptr; }
    vtkSmartPointer<vtkActor> loadImageActor(const cv::Mat&) override { return nullptr; }
    void showImage(const QString&) override {}
    void showImage(const cv::Mat&) override {}
    void setTextureActor(vtkSmartPointer<vtkActor>) override {}
    void setPointCloudActor(vtkSmartPointer<vtkActor>) override {}
    void clear3DModel() override {}
    void clear2DTexture() override {}
    void clearPointCloud() override {}
    void resetToSingleRenderer() override {}
    bool onLoadDicom(const QString&) override { return false; }
    vtkRenderer* renderer() override { return nullptr; }
    void render() override {}
    void setPointCloudData(const std::vector<Point3D>&, const std::vector<PointColor>&) override {}
    bool isPointCloudVisible() const override { return false; }
    void setPointCloudVisible(bool) override {}
    void applyViewSettings(const QString&) override {}
};

#endif
