#ifndef ISCENESERVICE_H
#define ISCENESERVICE_H

#include <QString>
#include <vector>
#include "PointCloudData.h"

// ABI boundary: plugin must not link OpenCV just to include this interface.
// Only cv::Mat remains (texture/image actor plumbing); point-cloud types are ABI-neutral.
namespace cv { class Mat; }

class vtkRenderer;
class vtkActor;
template <class T> class vtkSmartPointer;

/**
 * @brief Scene/render access for plugins.
 *
 * ACCEPTED EXCEPTION (documented deviation): this is the one interface that
 * intentionally exposes concrete VTK types (vtkRenderer*, vtkActor*,
 * vtkSmartPointer). Abstraction over VTK would add a full mirror hierarchy
 * for little safety gain, since every consumer already links VTK directly.
 * Keep new methods limited to actor/renderer plumbing — no vtk* algorithm
 * types (mappers, filters, volumes) beyond these three forms.
 *
 * The shell's QVTKOpenGLNativeWidget is deliberately NOT exposed: a plugin
 * gets actor/renderer plumbing, never the widget that owns the OpenGL surface
 * (replacing its interactor or painting over the shell is not a plugin right).
 */
class ISceneService {
public:
    /// ServiceRegistry lookup key (see ServiceRegistry.h).
    static constexpr const char *ServiceKey =
        "com.3dreconstruction.service.ISceneService";

    virtual ~ISceneService() = default;
    virtual vtkRenderer*            renderer() = 0;
    virtual void setTextureActor(vtkSmartPointer<vtkActor> actor) = 0;
    virtual void setPointCloudActor(vtkSmartPointer<vtkActor> actor) = 0;
    virtual void clear3DModel()    = 0;
    virtual void clear2DTexture()  = 0;
    virtual void clearPointCloud() = 0;
    virtual void resetToSingleRenderer() = 0;
    virtual void loadOBJwithMTL(const QString &objPath, const QString &mtlPath) = 0;
    /// Loads a DICOM series from an explicit directory. Returns false when the
    /// directory is empty or contains no readable series — the caller owns the
    /// file dialog and the error reporting (no UI in the service layer).
    virtual bool onLoadDicom(const QString &path) = 0;

    // ── Image display ────────────────────────────────────────────────────────
    /// Builds a texture actor from an image file / cv::Mat without showing it.
    /// The file/mat parsing stays inside the services layer; plugins never
    /// include loader internals.
    virtual vtkSmartPointer<vtkActor> loadImageActor(const QString& imagePath) = 0;
    virtual vtkSmartPointer<vtkActor> loadImageActor(const cv::Mat& image) = 0;
    /// Convenience: loads and displays the actor on the main renderer in one
    /// call (equivalent to setTextureActor(loadImageActor(...))).
    virtual void showImage(const QString& imagePath) = 0;
    virtual void showImage(const cv::Mat& image) = 0;

    virtual bool isPointCloudVisible() const = 0;
    virtual void setPointCloudVisible(bool visible) = 0;

    // ── Render & Point Cloud display ──────────────────────────────────────────
    /// Request the active render window to repaint.
    virtual void render() = 0;

    /**
     * @brief Set and render a 3D point cloud with optional RGB colors.
     * Manages internal VTK glyph/mapper pipelines and resets the camera.
     */
    virtual void setPointCloudData(const std::vector<Point3D>& points,
                                   const std::vector<PointColor>& colors = {}) = 0;

    /**
     * Áp dụng các thiết lập hiển thị từ UserManager (màu nền, trục toạ độ, lưới nền).
     * Gọi khi user nhấn "Áp dụng" trong Settings Dialog.
     */
    virtual void applyViewSettings(const QString& username) = 0;
};

#endif // ISCENESERVICE_H
