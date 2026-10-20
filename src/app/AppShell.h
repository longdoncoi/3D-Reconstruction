#ifndef APPSHELL_H
#define APPSHELL_H

#include "IAppContext.h"
#include "IShellService.h"
#include "IReconstructionService.h"
#include "IAIService.h"
#include "SignalBus.h"
#include "ServiceRegistry.h"
#include <QObject>
#include <memory>

class QVTKOpenGLNativeWidget;
class QMainWindow;
class QMenuBar;
class QMenu;
class QWidget;
class SceneService;
class ViewerService;
class SettingsService;
class ReconstructionService;
class AIService;
class AIModelLoadService;
class AIAssistant;
class MailService;
class SmtpOutbox;
class ShellService;
class LanguageManager;
class IProgressDialog;

/**
 * @brief Composition root — only file allowed to include concrete service
 * implementations (SceneService, ReconstructionService, etc.). All other
 * consumers must resolve via IAppContext / registry.
 */

/**
 * @brief Concrete implementation của IAppContext — tách biệt hoàn toàn khỏi MainWindow.
 *
 * AppShell:
 *  - Own toàn bộ services (SceneService, ViewerService, ..., AI, Reconstruction)
 *  - Đăng ký services vào ServiceRegistry để plugin look-up
 *  - MainWindow chỉ construct AppShell và delegate IAppContext calls
 *
 * Quy trình khởi tạo:
 *   m_shell = std::make_unique<AppShell>(this, m_vtkWidget, this);
 *   m_shell->initializeServices();
 */
class AppShell : public QObject, public IAppContext {
    Q_OBJECT

public:
    explicit AppShell(QMainWindow*           mainWindow,
                      QVTKOpenGLNativeWidget* vtkWidget,
                      QObject*               parent = nullptr);
    ~AppShell() override;

    /**
     * @brief Khởi tạo tất cả services và đăng ký vào ServiceRegistry.
     * Phải gọi sau khi constructor MainWindow hoàn thành (widget đã sẵn sàng).
     */
    void initializeServices();

    /**
     * @brief Phase-2 services (reconstruction/AI/mail — heavy, non-UI).
     * Idempotent; MainWindow calls this right before loadPlugins() so that
     * plugins always find a fully populated registry.
     */
    void initializeDeferredServices();

    // ─── IAppContext ───────────────────────────────────────────────────────────
    ISceneService*    scene()      override;
    IViewerService*   viewer()     override;
    ISettingsService* settings()   override;
    SignalBus*        signalBus()  override { return m_signalBus; }
    IShellService*    shell()      override;
    ServiceRegistry*  services()   override { return m_registry.get(); }

    QString   translate(const QString& key) override;
    void      setLanguage(const QString& lang) override;

    // ─── Setters dùng bởi MainWindow sau khi UI build xong (wiring, not API) ─
    void setMenuBar(QMenuBar* bar);
    void setProgressDialog(IProgressDialog* dlg);

private:
    // Injected from MainWindow (shell wiring only — IAppContext no longer
    // exposes the concrete QMainWindow; plugins use IShellService).
    QVTKOpenGLNativeWidget* m_vtkWidget;

    // Owned services
    SignalBus*                                  m_signalBus    = nullptr;
    std::unique_ptr<ServiceRegistry>            m_registry;
    std::unique_ptr<ShellService>               m_shellService;
    std::unique_ptr<SceneService>               m_sceneService;
    std::unique_ptr<ViewerService>              m_viewerService;
    std::unique_ptr<SettingsService>            m_settingsService;
    std::unique_ptr<ReconstructionService>      m_reconService;
    std::unique_ptr<AIService>                  m_aiService;
    /// Must be declared after AIService so it is destroyed before it: the
    /// model-load worker keeps a pointer to IAIService.
    std::unique_ptr<AIModelLoadService>         m_aiModelLoadService;
    std::unique_ptr<AIAssistant>                m_aiAssistant;
    std::unique_ptr<MailService>                m_mailService;
    std::unique_ptr<SmtpOutbox>                 m_outbox; ///< IOutgoingMail seam for UserManager
    std::unique_ptr<LanguageManager>            m_languageManager;
};

#endif // APPSHELL_H
