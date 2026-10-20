#ifndef PLUGINCONTEXTMOCK_H
#define PLUGINCONTEXTMOCK_H

#include "core/IAIAssistantService.h"
#include "core/IAIModelLoadService.h"
#include "core/IAIService.h"
#include "core/IAppContext.h"
#include "core/IAppPaths.h"
#include "core/ILanguageService.h"
#include "core/IMailService.h"
#include "core/IProgressDialog.h"
#include "core/IShellService.h"
#include "core/IUserManager.h"
#include "core/SignalBus.h"
#include "core/ServiceRegistry.h"
#include "core/ISceneService.h"
#include "core/ISettingsService.h"
#include "core/IViewerService.h"

#include "mock/IReconstructionServiceMock.h"

#include <vtkSmartPointer.h>
#include <vtkActor.h>

#include <QDockWidget>
#include <QHash>
#include <QJsonObject>
#include <QList>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QVBoxLayout>
#include <QStringList>
#include <QVariantMap>

#include <opencv2/core.hpp>

/**
 * @brief Stub implementations of the plugin-facing kernel interfaces.
 *
 * The plugin layer (src/modules/**) was the last unmeasured part of src/:
 * the shipped plugins are GUI shells, and the only test that touched them
 * (TestPluginSmoke) loaded their metadata without ever calling
 * initialize().  To measure the plugins' own code, the host has to hand them
 * a context — which is exactly what these stubs provide.
 *
 * The stubs answer "nothing to do": no server process, no model, no network,
 * no file system.  That keeps the test hermetic while still executing the
 * plugin's own wiring code (service lookups, dock/ribbon construction,
 * signal connections, teardown).
 */
namespace PluginMock {

// ── Progress dialog ──────────────────────────────────────────────────────────
class ProgressDialog : public IProgressDialog {
public:
    void setRange(int minimum, int maximum) override { m_min = minimum; m_max = maximum; }
    void setValue(int value) override { m_value = value; }
    void setLabelText(const QString &text) override { m_label = text; }
    void reset() override { m_value = m_min; }
    void centerOnWidget(QWidget *) override {}
    void show() override { m_visible = true; }
    void hide() override { m_visible = false; }

    int value() const { return m_value; }
    QString label() const { return m_label; }
    bool visible() const { return m_visible; }

private:
    int m_min = 0;
    int m_max = 0;
    int m_value = 0;
    bool m_visible = false;
    QString m_label;
};

// ── Shell service ────────────────────────────────────────────────────────────
class ShellService : public IShellService {
public:
    ShellService() : m_menuBar(nullptr) {
        // The real shell is a window: the dialogs and dock widgets the
        // plugins create are parented to it, so they die with the shell —
        // before the plugin DLL is unloaded.  The stub keeps that ownership
        // (a parent-less dock would outlive the DLL and crash the teardown).
        m_shellWidget.setCentralWidget(new QWidget);
        // The real shell's menu bar is a child of the main window, not a
        // top-level widget: plugins parent their buttons to it.
        m_menuBar = new QMenuBar(&m_shellWidget);
    }

    QMenu *getMenu(const QString &id) override {
        // The real shell creates the menu on demand and owns it; a plugin
        // dereferences the returned pointer without checking.
        auto it = m_menus.find(id);
        if (it == m_menus.end()) {
            QMenu *menu = m_menuBar->addMenu(id);
            m_menus.insert(id, menu);
            return menu;
        }
        return it.value();
    }
    QMenuBar *menuBar() override { return m_menuBar; }
    void updateMenuStates() override {}
    QWidget *getTabPanel(const QString &) override { return nullptr; }
    IProgressDialog *getProgressDialog() override { return &m_progress; }
    void registerTabPanel(const QString &, QWidget *) override {}
    void addDockWidget(Qt::DockWidgetArea area, QDockWidget *dock) override {
        Q_UNUSED(area); Q_UNUSED(dock);
    }
    void deleteDockWidget(QDockWidget *dock) override {
        // The real shell detaches the dock from the main window before
        // deleting it; skipping the detach leaves a stale item in the dock
        // area and the *next* plugin's addDockWidget() crashes.
        if (!dock) {
            return;
        }
        delete dock;
    }
    QWidget *dialogParent() override { return &m_shellWidget; }
    QWidget *centralWidget() override { return m_shellWidget.centralWidget(); }
    void setVisible(bool) override {}

    ProgressDialog &progressDialog() { return m_progress; }
    QWidget *shellWidget() { return &m_shellWidget; }

private:
    QMenuBar *m_menuBar = nullptr;
    QHash<QString, QMenu *> m_menus;
    QMainWindow m_shellWidget;
    ProgressDialog m_progress;
};

// ── Language service ─────────────────────────────────────────────────────────
class LanguageService : public ILanguageService {
public:
    QString currentLanguage() const override { return QStringLiteral("en"); }
    void setLanguage(const QString &) override {}
    QString translate(const QString &key) const override { return key; }
    QString translate(const char *key) const override { return QString::fromLatin1(key); }
    QString displayName() const override { return QStringLiteral("English"); }
    QString flagEmoji() const override { return QStringLiteral("EN"); }
};

// ── App paths ────────────────────────────────────────────────────────────────
class AppPaths : public IAppPaths {
public:
    explicit AppPaths(QString root) : m_root(std::move(root)) {}
    QString appDir() const override { return m_root; }
    QString projectRootDir() const override { return m_root; }
    QString dataRootDir() const override { return m_root; }
    QString configPath() const override { return m_root + "/Config.ini"; }
    QString logsDir() const override { return m_root + "/logs"; }
    QString modelsDir() const override { return m_root + "/Models"; }
    QString predictDir(const QString &type) const override { return m_root + "/" + type; }
    QString aiAssistantDir() const override { return m_root + "/AIAssistant"; }
    QString aiComputerVisionDir() const override { return m_root + "/AIComputerVision"; }
    QString uploadDir() const override { return m_root + "/Upload"; }
    QString thumbnailsDir() const override { return m_root + "/Thumbnails"; }
    QString pluginsDir() const override { return m_root + "/plugins"; }

private:
    QString m_root;
};

// ── User manager ─────────────────────────────────────────────────────────────
class UserManager : public IUserManager {
public:
    bool loadConfig(const QString &) override { return true; }
    void saveConfig() override {}
    bool login(const QString &, const QString &, QString &, AuthError *) override { return false; }
    void logout() override { m_loggedIn = false; }
    bool isLoggedIn() const override { return m_loggedIn; }
    UserInfo currentUser() const override { return m_user; }
    QString currentUsername() const override { return m_loggedIn ? m_user.username : QString(); }
    void setRememberMe(bool, const QString &) override {}
    bool isRememberMeEnabled() const override { return false; }
    QString savedUsername() const override { return {}; }
    bool changePassword(const QString &, const QString &, const QString &, QString &, AuthError *) override {
        return false;
    }
    bool resetPassword(const QString &, const QString &) override { return false; }
    void updateAvatar(const QString &, const QString &) override {}
    bool sendPasswordResetOtp(const QString &, QString &) override { return false; }
    bool verifyOtp(const QString &, const QString &) override { return false; }
    QString findUsernameByEmail(const QString &) const override { return {}; }
    QStringList allUsernames() const override { return {}; }
    UserInfo userInfo(const QString &) const override { return {}; }
    bool addUser(const UserInfo &, const QString &, QString &, AuthError *) override { return false; }
    bool updateUser(const UserInfo &, QString &) override { return false; }
    bool deleteUser(const QString &, QString &) override { return false; }
    QString smtpSenderEmail() const override { return {}; }
    QString smtpSenderPassword() const override { return {}; }
    void setSmtpCredentials(const QString &, const QString &) override {}
    QString configPath() const override { return {}; }

    // ILicenseService
    QString licenseStatusStr(const QString &) const override { return QStringLiteral("None"); }
    int daysRemaining(const QString &) const override { return 0; }
    bool needsActivation(const QString &) const override { return false; }
    bool isLicenseExpired(const QString &) const override { return false; }
    bool activateTrial(const QString &, QString &, AuthError *) override { return false; }
    bool activateLicenseKey(const QString &, const QString &, QString &, AuthError *) override {
        return false;
    }
    QString generateLicenseKey() override { return {}; }
    QStringList availableLicenseKeys() const override { return {}; }
    bool revokeLicenseKey(const QString &, QString &) override { return false; }

    // IUserPreferences
    QString getUserPref(const QString &, const QString &, const QString &) const override { return {}; }
    void setUserPref(const QString &, const QString &, const QString &) override {}
    QString getUserSecret(const QString &, const QString &, const QString &) const override { return {}; }
    void setUserSecret(const QString &, const QString &, const QString &) override {}

    /// Pretend a user is signed in: the auth plugin then skips its login
    /// dialog and exercises the rest of its initialization.
    void signIn(const QString &username) {
        m_loggedIn = true;
        m_user.username = username;
    }

private:
    bool m_loggedIn = false;
    UserInfo m_user;
};

// ── AI assistant ─────────────────────────────────────────────────────────────
class AIAssistantService : public IAIAssistantService {
public:
    void startServer(int) override {}
    void stopServer() override {}
    void sendMessage(const QString &, const QStringList &) override {}
    void sendMessageToSession(const QString &, const QString &, const QStringList &) override {}
    void retryMessage(const QString &, int) override {}
    void retryAgentTask(const QString &, int) override {}
    void editMessage(const QString &, int, const QString &) override {}
    void switchModel(int) override {}
    void restartModel() override {}
    void restartRAG() override {}
    void restartAgent() override {}
    void restartServer() override {}
    void newChat() override {}
    void loadSession(const QString &) override {}
    void deleteSession(const QString &) override {}
    void clearHistory() override {}
    void reloadSessions() override {}
    QList<ChatSession> getSessions() const override { return {}; }
    QString currentSessionId() const override { return {}; }
    QList<QJsonObject> getHistory() const override { return {}; }
    bool isThinking() const override { return false; }
    bool isSessionThinking(const QString &) const override { return false; }
    int sessionThinkingInsertIndex(const QString &) const override { return -1; }
    bool isServerRunning() const override { return false; }
    void executeAgentTask(const QString &, const QString &, const QStringList &) override {}
    void approveAgentAction(const QString &, const QString &) override {}
    void rejectAgentAction(const QString &, const QString &) override {}
    void reportUiActionResult(const QString &, bool, const QVariantMap &) override {}
};

// ── AI service (vision) ──────────────────────────────────────────────────────
class AIService : public IAIService {
public:
    bool loadDetectionModel(const QString &) override { return false; }
    bool loadSegmentationModel(const QString &) override { return false; }
    bool loadTrackingModel(const QString &) override { return false; }
    bool isDetectionReady() const override { return false; }
    bool isSegmentationReady() const override { return false; }
    bool isTrackingReady() const override { return false; }
    std::vector<AIDetectionResult> detectObjects(const cv::Mat &) override { return {}; }
    cv::Mat runDetection(const cv::Mat &) override { return {}; }
    cv::Mat runSegmentation(const cv::Mat &) override { return {}; }
    cv::Mat runTracking(const cv::Mat &) override { return {}; }
    void resetTrackingState() override {}
    void setConfidenceThreshold(float threshold) override { m_threshold = threshold; }
    float confidenceThreshold() const override { return m_threshold; }

private:
    float m_threshold = 0.5f;
};

// ── AI model loader ──────────────────────────────────────────────────────────
class AIModelLoadService : public IAIModelLoadService {
public:
    void loadModelsAsync(const QString &, const QString &, const QString &) override {}
    void cancel() override {}
    bool isLoading() const override { return false; }
};

// ── Mail service ─────────────────────────────────────────────────────────────
class MailService : public IMailService {
public:
    bool sendMail(const MailMessage &, QString &errorMsg) override {
        errorMsg = QStringLiteral("stub");
        return false;
    }
    QList<MailMessage> fetchInbox(int, QString &errorMsg) override {
        errorMsg = QStringLiteral("stub");
        return {};
    }
    bool markRead(const QString &, QString &errorMsg) override {
        errorMsg = QStringLiteral("stub");
        return false;
    }
    bool deleteMail(const QString &, QString &errorMsg) override {
        errorMsg = QStringLiteral("stub");
        return false;
    }
    void setCredentials(const QString &, const QString &, const QString &) override {}
    bool testConnection(QString &errorMsg) override {
        errorMsg = QStringLiteral("stub");
        return false;
    }
    bool hasCredentials() const override { return false; }
    QString senderEmail() const override { return {}; }
};

// ── Scene / viewer / settings ────────────────────────────────────────────────
class SceneService : public ISceneService {
public:
    vtkSmartPointer<vtkActor> loadImageActor(const QString &) override { return nullptr; }
    vtkSmartPointer<vtkActor> loadImageActor(const cv::Mat &) override { return nullptr; }
    void showImage(const QString &) override {}
    void showImage(const cv::Mat &) override {}
    void setTextureActor(vtkSmartPointer<vtkActor>) override {}
    void setPointCloudActor(vtkSmartPointer<vtkActor>) override {}
    void clear3DModel() override {}
    void clear2DTexture() override {}
    void clearPointCloud() override {}
    void resetToSingleRenderer() override {}
    void loadOBJwithMTL(const QString &, const QString &) override {}
    bool onLoadDicom(const QString &) override { return false; }
    vtkRenderer *renderer() override { return nullptr; }
    QVTKOpenGLNativeWidget *vtkWidget() override { return nullptr; }
    void render() override {}
    void setPointCloudData(const std::vector<Point3D> &, const std::vector<PointColor> &) override {}
    bool isPointCloudVisible() const override { return false; }
    void setPointCloudVisible(bool) override {}
    void applyViewSettings(const QString &) override {}
};

class ViewerService : public IViewerService {
public:
    QString getCurrent2DImagePath() const override { return {}; }
    void setCurrent2DImagePath(const QString &) override {}
    void setImageList(const QStringList &, int) override {}
    void loadCurrentIndexImage() override {}
    void onNextImage() override {}
    void onPrevImage() override {}
    void onAutoNext() override {}
    void onAutoPrev() override {}
    AIMode getCurrentAIMode() const override { return AIMode::None; }
    void setAIMode(AIMode) override {}
    bool isUpdateSceneEnabled() const override { return false; }
    void setUpdateSceneEnabled(bool) override {}
};

class SettingsService : public ISettingsService {
public:
    QString getLastUsedPath(const QString &) const override { return {}; }
    void setLastUsedPath(const QString &, const QString &) override {}
};

// ── Application context ──────────────────────────────────────────────────────
/**
 * The context handed to the plugins.  It owns the stub services, so they stay
 * alive for the whole plugin lifetime — the same ordering guarantee the real
 * composition root gives.
 */
class AppContext : public IAppContext {
public:
    explicit AppContext(const QString &root)
        : m_paths(root) {
        m_registry.registerService<ILanguageService>(&m_language);
        m_registry.registerService<IUserManager>(&m_users);
        m_registry.registerService<IAppPaths>(&m_paths);
        m_registry.registerService<IAIAssistantService>(&m_assistant);
        m_registry.registerService<IAIService>(&m_ai);
        m_registry.registerService<IAIModelLoadService>(&m_loader);
        m_registry.registerService<IMailService>(&m_mail);
        m_registry.registerService<IReconstructionService>(&m_reconstruction);
    }

    ISceneService *scene() override { return &m_scene; }
    IViewerService *viewer() override { return &m_viewer; }
    ISettingsService *settings() override { return &m_settings; }
    SignalBus *signalBus() override { return &m_signalBus; }
    IShellService *shell() override { return &m_shell; }
    ServiceRegistry *services() override { return &m_registry; }
    QString translate(const QString &key) override { return key; }
    void setLanguage(const QString &) override {}

    UserManager &users() { return m_users; }
    AppPaths &paths() { return m_paths; }
    ShellService &shellService() { return m_shell; }

private:
    ServiceRegistry m_registry;
    SignalBus m_signalBus;
    LanguageService m_language;
    AppPaths m_paths;
    UserManager m_users;
    AIAssistantService m_assistant;
    AIService m_ai;
    AIModelLoadService m_loader;
    MailService m_mail;
    MockReconstructionService m_reconstruction;
    SceneService m_scene;
    ViewerService m_viewer;
    SettingsService m_settings;
    ShellService m_shell;
};

} // namespace PluginMock

#endif // PLUGINCONTEXTMOCK_H
