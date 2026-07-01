#ifndef APPSHELL_H
#define APPSHELL_H

#include "IAppContext.h"
#include "IReconstructionService.h"
#include "IAIService.h"
#include "SignalBus.h"
#include "ServiceRegistry.h"
#include <QObject>
#include <memory>

class QVTKOpenGLNativeWidget;
class QMenuBar;
class QMenu;
class QWidget;
class SceneService;
class ViewerService;
class SettingsService;
class ReconstructionService;
class AIService;
class AIAssistant;
class MailService;

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

    // ─── IAppContext ───────────────────────────────────────────────────────────
    ISceneService*    scene()      override;
    IViewerService*   viewer()     override;
    ISettingsService* settings()   override;
    QMainWindow*      mainWindow() override { return m_mainWindow; }
    SignalBus*        signalBus()  override { return m_signalBus; }
    ServiceRegistry*  services()   override { return m_registry.get(); }

    CustomProgressDialog*   getProgressDialog() override { return m_progressDialog; }

    QMenu*    getMenu(const QString& id) override;
    QMenuBar* menuBar() override;
    void      updateMenuStates() override;
    QWidget*  getTabPanel(const QString& tabName) override;
    QString   translate(const QString& key) override;
    void      setLanguage(const QString& lang) override;

    // ─── Setters dùng bởi MainWindow sau khi UI build xong ───────────────────
    void setMenuBar(QMenuBar* bar)        { m_menuBar = bar; }
    void setProgressDialog(CustomProgressDialog* dlg) { m_progressDialog = dlg; }
    void registerTabPanel(const QString& key, QWidget* panel);

private:
    // Injected from MainWindow
    QMainWindow*            m_mainWindow;
    QVTKOpenGLNativeWidget* m_vtkWidget;
    QMenuBar*               m_menuBar       = nullptr;
    CustomProgressDialog*   m_progressDialog = nullptr;

    // Owned services
    SignalBus*                                  m_signalBus    = nullptr;
    std::unique_ptr<ServiceRegistry>            m_registry;
    std::unique_ptr<SceneService>               m_sceneService;
    std::unique_ptr<ViewerService>              m_viewerService;
    std::unique_ptr<SettingsService>            m_settingsService;
    std::unique_ptr<ReconstructionService>      m_reconService;
    std::unique_ptr<AIService>                  m_aiService;
    std::unique_ptr<AIAssistant>                m_aiAssistant;
    std::unique_ptr<MailService>                m_mailService;

    // Tab panels map (key → QWidget*)
    QMap<QString, QWidget*> m_tabPanels;
    
    // Cached menus
    QMap<QString, QMenu*> m_menus;
};

#endif // APPSHELL_H
