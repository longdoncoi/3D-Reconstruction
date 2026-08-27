#ifndef SCENESERVICE_H
#define SCENESERVICE_H

#include "IAppContext.h"
#include "ISceneService.h"
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <QVTKOpenGLNativeWidget.h>
#include <vector>
#include "CrosshairManager.h"

class SceneService : public QObject, public ISceneService {
    Q_OBJECT
public:
    SceneService(IAppContext* ctx, QVTKOpenGLNativeWidget* vtkWidget, QObject* parent = nullptr);
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
    void onLoadDicom(const QString &path = "") override;
    
    bool isPointCloudVisible() const override { return m_pointCloudVisible; }
    void setPointCloudVisible(bool visible) override { m_pointCloudVisible = visible; }
    void applyViewSettings(const QString& username) override;

private:
    void setupDicomRenderers(vtkSmartPointer<vtkImageData> volume);
    void setupCrosshairInteractor();

    IAppContext* m_ctx;
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
