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
class vtkRenderWindow;

/**
 * @brief Documented exception (ISceneService): intentionally inherits QObject
 * separately because it must emit Qt signals/meta-object for VTK widget
 * wiring. No diamond: ISceneService is plain (no QObject base).
 */
class APP_EXPORT SceneService : public QObject, public ISceneService {
    Q_OBJECT
public:
    /// @param renderWindow Optional explicit render window. When null, the
    ///        window is taken from @p vtkWidget. The shell always passes a
    ///        widget; the parameter is a real seam, not a test hook: the
    ///        service then depends on a render window instead of a concrete
    ///        QVTK widget, and can be exercised without an OpenGL surface.
    SceneService(IUserPreferences* prefs = nullptr,
                 QVTKOpenGLNativeWidget* vtkWidget = nullptr,
                 QObject* parent = nullptr,
                 vtkRenderWindow* renderWindow = nullptr);
    ~SceneService();

    vtkRenderer*            renderer() override { return m_renderer; }
    
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

    /// The window the service renders into: the explicit one when injected,
    /// otherwise the one owned by the QVTK widget (null in headless use).
    vtkRenderWindow* renderWindow() const;

    IUserPreferences* m_prefs = nullptr; ///< not owned (AppShell); may be null in tests
    QVTKOpenGLNativeWidget* m_vtkWidget;
    vtkRenderWindow* m_renderWindow = nullptr; ///< not owned; may be null
    vtkSmartPointer<vtkRenderer> m_renderer;
    vtkSmartPointer<vtkRenderer> m_axialRenderer, m_sagittalRenderer, m_coronalRenderer;
    
    vtkSmartPointer<vtkActor> m_cloudActor;
    std::vector<vtkSmartPointer<vtkActor>> m_modelActors;
    vtkSmartPointer<vtkActor> m_texturePlaneActor;
    
    bool m_pointCloudVisible = false;

    // View settings actors
    vtkSmartPointer<vtkActor> m_gridActor;
    vtkSmartPointer<class vtkOrientationMarkerWidget> m_axesWidget;

    CrosshairManager* m_crosshair = nullptr;
    vtkSmartPointer<CrosshairInteractorStyle> m_crosshairStyle;
};

#endif // SCENESERVICE_H
