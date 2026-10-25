#ifndef SCENESERVICE_H
#define SCENESERVICE_H

#include "ISceneService.h"
#include "Global.h"
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <QVTKOpenGLNativeWidget.h>
#include <vector>
#include "CrosshairManager.h"

class IUserPreferences;

/**
 * @brief Documented exception (ISceneService): intentionally inherits QObject
 * separately because it must emit Qt signals/meta-object for VTK widget
 * wiring. No diamond: ISceneService is plain (no QObject base).
 */
class APP_EXPORT SceneService : public QObject, public ISceneService {
    Q_OBJECT
public:
    SceneService(IUserPreferences* prefs = nullptr, QVTKOpenGLNativeWidget* vtkWidget = nullptr, QObject* parent = nullptr);
    ~SceneService();

    vtkRenderer*            renderer() override { return m_renderer; }
    QVTKOpenGLNativeWidget* vtkWidget() override { return m_vtkWidget; }
    
    void setTextureActor(vtkSmartPointer<vtkActor> actor) override;
    void setPointCloudActor(vtkSmartPointer<vtkActor> actor) override;
    void clear3DModel() override;
    void clear2DTexture() override;
    void clearPointCloud() override;
    void resetToSingleRenderer() override;
    void loadOBJwithMTL(const QString &objPath, const QString &mtlPath) override;
    bool onLoadDicom(const QString &path) override;

    vtkSmartPointer<vtkActor> loadImageActor(const QString& imagePath) override;
    vtkSmartPointer<vtkActor> loadImageActor(const cv::Mat& image) override;
    void showImage(const QString& imagePath) override;
    void showImage(const cv::Mat& image) override;

    bool isPointCloudVisible() const override { return m_pointCloudVisible; }
    void setPointCloudVisible(bool visible) override { m_pointCloudVisible = visible; }
    void render() override;
    void setPointCloudData(const std::vector<Point3D>& points,
                           const std::vector<PointColor>& colors = {}) override;
    void applyViewSettings(const QString& username) override;

private:
    void setupDicomRenderers(vtkSmartPointer<vtkImageData> volume);
    void setupCrosshairInteractor();

    IUserPreferences* m_prefs = nullptr; ///< not owned (AppShell); may be null in tests
    QVTKOpenGLNativeWidget* m_vtkWidget;
    vtkSmartPointer<vtkRenderer> m_renderer;
    vtkSmartPointer<vtkRenderer> m_axialRenderer, m_sagittalRenderer, m_coronalRenderer;
    
    vtkSmartPointer<vtkActor> m_cloudActor;
    std::vector<vtkSmartPointer<vtkActor>> m_modelActors;
    vtkSmartPointer<vtkActor> m_texturePlaneActor;
    
    bool m_pointCloudVisible = false;

    // View settings actors
    vtkSmartPointer<vtkActor> m_gridActor;
    class vtkOrientationMarkerWidget* m_axesWidget = nullptr;

    CrosshairManager* m_crosshair = nullptr;
    vtkSmartPointer<CrosshairInteractorStyle> m_crosshairStyle;
};

#endif // SCENESERVICE_H
