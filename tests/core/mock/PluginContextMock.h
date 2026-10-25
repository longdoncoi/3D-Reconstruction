#ifndef PLUGINCONTEXTMOCK_H
#define PLUGINCONTEXTMOCK_H

#include "include/IAIAssistantService.h"
#include "include/IAIModelLoadService.h"
#include "include/IAIService.h"
#include "include/IAppContext.h"
#include "include/IAppPaths.h"
#include "include/ILanguageService.h"
#include "include/IMailService.h"
#include "include/IProgressDialog.h"
#include "include/IShellService.h"
#include "include/IUserManager.h"
#include "include/SignalBus.h"
#include "include/ServiceRegistry.h"
#include "core/MutableServiceRegistry.h"
#include "include/ISceneService.h"
#include "include/ISettingsService.h"
#include "include/IViewerService.h"

#include "mock/IReconstructionServiceMock.h"

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkRenderer.h>

#include <QDockWidget>
#include <QHash>
#include <QJsonObject>
#include <QList>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QThread>
#include <QVBoxLayout>
#include <QStringList>
#include <QVariantMap>

#include <opencv2/core.hpp>

#include <stdexcept>

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
/// Counts the children a plugin creates in a tab panel. It makes the
/// widget-ownership check non-vacuous: a plugin that installs nothing must
/// not be able to "pass" by never touching the shell.
class ChildWatcher : public QObject {
public:
    using QObject::QObject;

protected:
    bool eventFilter(QObject *obj, QEvent *event) override
    {
        if (event->type() == QEvent::ChildAdded) ++count;
        return QObject::eventFilter(obj, event);
    }

public:
    int count = 0;
};

class ShellService : public IShellService {
public:
    ShellService() : m_menuBar(nullptr) {
        // The real shell is a window: the dialogs and dock widgets the
        // plugins create are parented to it, so they die with the shell —
        // before the plugin DLL is unloaded.  The stub keeps that ownership
        // (a parent-less dock would outlive the DLL and crash the teardown).
        // The central widget carries a layout exactly like MainWindow's:
        // ViewerListUI only installs its list when the central widget has a
        // QVBoxLayout, so without one the plugin's list path is invisible to
        // the test.
        auto *central = new QWidget;
        central->setLayout(new QVBoxLayout);
        m_shellWidget.setCentralWidget(central);
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
    QWidget *getTabPanel(const QString &tabName) override {
        // The real shell creates one panel per tab and plugins add their
        // ribbon group to it. Returning a REAL panel (with the same
        // QHBoxLayout the shell uses) is what makes the widget-ownership
        // contract observable: a plugin that forgets to remove its group
        // leaves a child behind, and the smoke test sees it.
        auto it = m_tabPanels.find(tabName);
        if (it == m_tabPanels.end()) {
            QWidget *panel = new QWidget(&m_shellWidget);
            QHBoxLayout *layout = new QHBoxLayout(panel);
            layout->setContentsMargins(4, 2, 4, 2);
            layout->setSpacing(4);
            layout->addStretch();
            ChildWatcher *watcher = new ChildWatcher(panel);
            panel->installEventFilter(watcher);
            m_panelWatchers.append(watcher);
            it = m_tabPanels.insert(tabName, panel);
        }
        return it.value();
    }
    IProgressDialog *getProgressDialog() override { return &m_progress; }
    void registerTabPanel(const QString &, QWidget *) override {}
    void addDockWidget(Qt::DockWidgetArea area, QDockWidget *dock) override {
        if (!dock) return;
        m_addedDocks.append(dock);
        ++m_installedDocks;
        m_shellWidget.addDockWidget(area, dock);
    }
    void deleteDockWidget(QDockWidget *dock) override {
        // The real shell detaches the dock from the main window before
        // deleting it; skipping the detach leaves a stale item in the dock
        // area and the *next* plugin's addDockWidget() crashes.
        if (!dock) {
            return;
        }
        m_shellWidget.removeDockWidget(dock);
        m_addedDocks.removeAll(dock);
        delete dock;
    }
    QWidget *dialogParent() override { return &m_shellWidget; }
    QWidget *centralWidget() override { return m_shellWidget.centralWidget(); }
    void setVisible(bool) override {}

    /**
     * The plugin-installed objects still living in the shell's widget tree.
     *
     * A plugin that releases everything leaves nothing behind. The shell's
     * own objects (the central widget, its menus) are not listed — only what
     * a plugin installed: the docks it added, the groups it created in a tab
     * panel and the tool buttons it put into the menu bar.
     */
    QStringList leftoverShellObjects() const
    {
        QStringList leftovers;
        for (QDockWidget *dock : m_addedDocks)
            leftovers << QStringLiteral("dock:%1").arg(dock->objectName());
        // A QAction added to a shell menu is re-parented to that menu, so it
        // outlives the plugin exactly like a widget does. Qt drops the
        // connection when the receiver dies, so this cannot crash — but it is
        // still an entry the plugin installed and never took back: the menu
        // keeps a dead item. The shell's own menus start empty, so every
        // non-separator action in them belongs to a plugin.
        for (auto it = m_menus.constBegin(); it != m_menus.constEnd(); ++it) {
            for (QAction *action : it.value()->actions()) {
                if (action->isSeparator() || action->menu()) continue;
                leftovers << QStringLiteral("menu:%1/%2").arg(it.key(), action->text());
            }
        }
        for (auto it = m_tabPanels.constBegin(); it != m_tabPanels.constEnd(); ++it) {
            const QWidgetList children = it.value()->findChildren<QWidget*>();
            for (QWidget *child : children)
                leftovers << QStringLiteral("panel:%1/%2").arg(it.key(), child->objectName());
        }
        const QWidgetList menuChildren = m_menuBar->findChildren<QWidget*>();
        for (QWidget *child : menuChildren) {
            // The shell's own menus belong to the shell, and Qt's own menu bar
            // internals (the extension button) are not a plugin's business;
            // a plugin's tool buttons are.
            if (qobject_cast<QMenu*>(child)) continue;
            if (child->objectName().startsWith(QStringLiteral("qt_"))) continue;
            leftovers << QStringLiteral("menubar:%1").arg(child->objectName());
        }
        return leftovers;
    }

    /**
     * The number of objects the plugin installed into the shell.
     *
     * Together with leftoverShellObjects() it shows that the ownership check
     * is not vacuous: a plugin that installs nothing would have nothing to
     * release either.
     */
    int installedShellObjectCount() const
    {
        int count = m_installedDocks;
        for (const ChildWatcher *watcher : m_panelWatchers) count += watcher->count;
        const QWidgetList menuChildren = m_menuBar->findChildren<QWidget*>();
        for (QWidget *child : menuChildren) {
            if (qobject_cast<QMenu*>(child)) continue;
            if (child->objectName().startsWith(QStringLiteral("qt_"))) continue;
            ++count;
        }
        return count;
    }

    ProgressDialog &progressDialog() { return m_progress; }
    QWidget *shellWidget() { return &m_shellWidget; }

private:
    QMenuBar *m_menuBar = nullptr;
    QHash<QString, QMenu *> m_menus;
    QHash<QString, QWidget *> m_tabPanels;
    QList<QDockWidget *> m_addedDocks;
    QList<ChildWatcher *> m_panelWatchers;
    int m_installedDocks = 0;
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
    bool resetPassword(const QString &, const QString &, QString &, AuthError *) override { return false; }
    bool updateProfile(const QString &, const QString &, const QString &, QString &, AuthError *) override {
        return false;
    }
    bool sendPasswordResetOtp(const QString &, QString &) override { return false; }
    bool verifyOtp(const QString &, const QString &) override { return false; }
    QString findUsernameByEmail(const QString &) const override { return {}; }
    bool isAdmin() const override { return m_admin; }
    QStringList allUsernames(AuthError *) const override { return {}; }
    UserInfo userInfo(const QString &, AuthError *) const override { return {}; }
    bool addUser(const UserInfo &, const QString &, QString &, AuthError *) override { return false; }
    bool updateUser(const UserInfo &, QString &, AuthError *) override { return false; }
    bool deleteUser(const QString &, QString &, AuthError *) override { return false; }
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
    QString generateLicenseKey(AuthError *) override { return {}; }
    QStringList availableLicenseKeys(AuthError *) const override { return {}; }
    bool revokeLicenseKey(const QString &, QString &, AuthError *) override { return false; }

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

    /// Pretend the signed-in user holds the administrator role.
    void signInAsAdmin(const QString &username = QStringLiteral("admin")) {
        signIn(username);
        m_user.role = UserRole::Admin;
        m_admin = true;
    }

private:
    bool m_loggedIn = false;
    bool m_admin = false;
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
    void renameSession(const QString &, const QString &) override {}
    void clearHistory() override {}
    void reloadSessions() override {}
    QList<ChatSession> getSessions() const override { return {}; }
    QString currentSessionId() const override { return {}; }
    QList<ChatMessage> getHistory() const override { return {}; }
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
    /// Test knobs: the plugin reads readiness, the inference itself records
    /// the thread it ran on so a test can prove it left the GUI thread.
    bool detectionReady = false;
    bool segmentationReady = false;
    bool trackingReady = false;
    bool throwOnInference = false;
    /// The production failure shape: AIService contains the exception and
    /// returns an EMPTY image (never throws). The plugin must treat that as a
    /// failure, not as "detected nothing".
    bool returnEmpty = false;
    int inferenceDelayMs = 0;
    int detectionCalls = 0;
    int segmentationCalls = 0;
    Qt::HANDLE lastInferenceThread = nullptr;

    bool isDetectionReady() const override { return detectionReady; }
    bool isSegmentationReady() const override { return segmentationReady; }
    bool isTrackingReady() const override { return trackingReady; }
    std::vector<AIDetectionResult> detectObjects(const cv::Mat &) override { return {}; }
    cv::Mat runDetection(const cv::Mat &) override {
        lastInferenceThread = QThread::currentThreadId();
        if (inferenceDelayMs > 0) QThread::msleep(static_cast<unsigned long>(inferenceDelayMs));
        if (throwOnInference) throw std::runtime_error("the inference backend failed");
        ++detectionCalls;
        if (returnEmpty) return {};
        return cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 255, 0));
    }
    cv::Mat runSegmentation(const cv::Mat &) override {
        lastInferenceThread = QThread::currentThreadId();
        if (inferenceDelayMs > 0) QThread::msleep(static_cast<unsigned long>(inferenceDelayMs));
        if (throwOnInference) throw std::runtime_error("the inference backend failed");
        ++segmentationCalls;
        if (returnEmpty) return {};
        return cv::Mat(8, 8, CV_8UC3, cv::Scalar(255, 0, 0));
    }
    cv::Mat runTracking(const cv::Mat &) override {
        lastInferenceThread = QThread::currentThreadId();
        if (inferenceDelayMs > 0) QThread::msleep(static_cast<unsigned long>(inferenceDelayMs));
        if (throwOnInference) throw std::runtime_error("the inference backend failed");
        if (returnEmpty) return {};
        return cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 255));
    }
    void resetTrackingState() override {}
    void setConfidenceThreshold(float threshold) override { m_threshold = threshold; }
    float confidenceThreshold() const override { return m_threshold; }

private:
    float m_threshold = 0.5f;
};

/// An AI service whose inference THROWS — the stand-in for a broken ONNX
/// session, a failing OpenCV call or a bad_alloc. Used to prove that a
/// caller on a background thread contains the exception instead of letting
/// it reach std::terminate() (which would abort the whole application).
class ThrowingAIService : public AIService {
public:
    cv::Mat runTracking(const cv::Mat &) override {
        throw std::runtime_error("the inference backend failed");
    }
    cv::Mat runSegmentation(const cv::Mat &) override {
        throw std::runtime_error("the inference backend failed");
    }
    cv::Mat runDetection(const cv::Mat &) override {
        throw std::runtime_error("the inference backend failed");
    }
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
        ++inboxCalls;
        errorMsg = inboxError;
        return inbox;
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

    /// What the next fetchInbox() returns. A dock that renders the inbox can
    /// be driven to both outcomes without a socket.
    QList<MailMessage> inbox;
    QString inboxError;
    int inboxCalls = 0;
};

// ── Scene / viewer / settings ────────────────────────────────────────────────
class SceneService : public ISceneService {
public:
    /// Knobs: an image load that returns an actor exercises the actor branch
    /// of ViewerViewModel (texture + camera reset), and the DICOM result
    /// decides whether the view model shows the modal error — a dialog the
    /// test cannot dismiss without an event loop.
    vtkSmartPointer<vtkActor> imageActor;
    bool dicomSucceeds = false;
    int dicomCalls = 0;
    QString lastObjPath;

    vtkSmartPointer<vtkActor> loadImageActor(const QString &) override { return imageActor; }
    vtkSmartPointer<vtkActor> loadImageActor(const cv::Mat &) override { return nullptr; }
    void showImage(const QString &) override {}
    void showImage(const cv::Mat &) override {}
    void setTextureActor(vtkSmartPointer<vtkActor>) override {}
    void setPointCloudActor(vtkSmartPointer<vtkActor>) override {}
    void clear3DModel() override {}
    void clear2DTexture() override {}
    void clearPointCloud() override {}
    void resetToSingleRenderer() override {}
    void loadOBJwithMTL(const QString &path, const QString &) override { lastObjPath = path; }
    bool onLoadDicom(const QString &) override { ++dicomCalls; return dicomSucceeds; }
    vtkRenderer *renderer() override { return m_renderer; }
    QVTKOpenGLNativeWidget *vtkWidget() override { return nullptr; }
    void render() override {}
    void setPointCloudData(const std::vector<Point3D> &, const std::vector<PointColor> &) override {}
    bool isPointCloudVisible() const override { return false; }
    void setPointCloudVisible(bool) override {}
    void applyViewSettings(const QString &) override {}

private:
    vtkSmartPointer<vtkRenderer> m_renderer = vtkSmartPointer<vtkRenderer>::New();
};

class ViewerService : public IViewerService {
public:
    QString getCurrent2DImagePath() const override { return m_currentPath; }
    void setCurrent2DImagePath(const QString &path) override { m_currentPath = path; }
    void setImageList(const QStringList &images, int index) override
    {
        m_imageList = images;
        m_imageIndex = index;
        ++setImageListCalls;
    }
    void loadCurrentIndexImage() override { ++loadCurrentIndexCalls; }
    void onNextImage() override {}
    void onPrevImage() override {}
    void onAutoNext() override {}
    void onAutoPrev() override {}
    AIMode getCurrentAIMode() const override { return m_mode; }
    void setAIMode(AIMode mode) override { m_mode = mode; }
    bool isUpdateSceneEnabled() const override { return false; }
    void setUpdateSceneEnabled(bool) override {}

    /// What the view model published, and how often the viewer was asked to
    /// load — the observable half of the ViewerListUI/ViewerViewModel contract.
    QStringList imageList() const { return m_imageList; }
    int imageIndex() const { return m_imageIndex; }
    int setImageListCalls = 0;
    int loadCurrentIndexCalls = 0;

private:
    QString m_currentPath;
    QStringList m_imageList;
    int m_imageIndex = -1;
    AIMode m_mode = AIMode::None;
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
        // The segregated contracts, exactly like the composition root.
        m_registry.registerService<IAuthService>(&m_users);
        m_registry.registerService<IUserProfileService>(&m_users);
        m_registry.registerService<IUserAdminService>(&m_users);
        m_registry.registerService<ISmtpCredentialService>(&m_users);
        m_registry.registerService<IUserPreferences>(&m_users);
        m_registry.registerService<ILicenseService>(&m_users);
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
    const ServiceRegistry *services() override { return &m_registry; }
    QString translate(const QString &key) override { return key; }
    void setLanguage(const QString &) override {}

    UserManager &users() { return m_users; }
    AppPaths &paths() { return m_paths; }
    ShellService &shellService() { return m_shell; }
    AIService &ai() { return m_ai; }
    SceneService &sceneService() { return m_scene; }
    ViewerService &viewerService() { return m_viewer; }
    MailService &mailService() { return m_mail; }

private:
    MutableServiceRegistry m_registry;
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
