#include "AppShell.h"
#include "MutableServiceRegistry.h"
#include "ISceneService.h"
#include "IUserManager.h"
#include "IAuthService.h"
#include "IUserProfileService.h"
#include "ISmtpCredentialService.h"
#include "IUserPreferences.h"
#include "ILicenseService.h"
#include "IViewerService.h"
#include "ISettingsService.h"
#include "IAppPaths.h"
#include "SceneService.h"
#include "ViewerService.h"
#include "SettingsService.h"
#include "ReconstructionService.h"
#include "AIService.h"
#include "AIModelLoadService.h"
#include "IReconstructionService.h"
#include "IAIService.h"
#include "IAIAssistantService.h"
#include "IMailService.h"
#include "MailService.h"
#include "SmtpOutbox.h"
#include "UserManager.h"
#include "LanguageManager.h"
#include "ILanguageService.h"
#include "AppConfig.h"
#include "Logger.h"
#include "IProgressDialog.h"
#include "ShellService.h"
#include "AIProcessor.h"
#include "AIAssistant.h"
#include "AIPredictionLogService.h"
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QWidget>
#include <QTimer>
#include <QVTKOpenGLNativeWidget.h>

AppShell::AppShell(QMainWindow*           mainWindow,
                   QVTKOpenGLNativeWidget* vtkWidget,
                   QObject*               parent)
    : QObject(parent)
    , m_vtkWidget(vtkWidget)
    , m_registry(std::make_unique<MutableServiceRegistry>())
    , m_languageManager(std::make_unique<LanguageManager>())
{
    m_signalBus = new SignalBus(this);

    // Widget-independent singletons register at CONSTRUCTION, not in
    // initializeServices(): MainWindow's UI setup (title bar / tab labels)
    // calls context()->translate() BEFORE initializeServices() runs, and
    // translate() must never see an empty registry. UserManager::instance()
    // is touched ONLY here (composition root).
    m_registry->registerService<IUserManager>(UserManager::instance());
    // The segregated contracts: the same instance under every narrow key, so
    // a consumer can depend on the ONE capability it needs instead of the
    // aggregate (Interface Segregation — see IUserManager.h).
    m_registry->registerService<IAuthService>(UserManager::instance());
    m_registry->registerService<IUserProfileService>(UserManager::instance());
    m_registry->registerService<IUserAdminService>(UserManager::instance());
    m_registry->registerService<ISmtpCredentialService>(UserManager::instance());
    m_registry->registerService<IUserPreferences>(UserManager::instance());
    m_registry->registerService<ILicenseService>(UserManager::instance());
    m_registry->registerService<ILanguageService>(m_languageManager.get());
    // Path provider — AppConfig remains the process singleton for bootstrap
    // (Main/Logger), but consumers resolve it through the registry so the
    // path surface is mockable and the singleton stays at the composition root.
    m_registry->registerService<IAppPaths>(&AppConfig::bootstrap());
    Logger::setLogsDir(AppConfig::bootstrap().logsDir());
    // UI-shell contract (menus/tabs/progress dialog) — implementation lives in
    // AppCore (ShellService) so the contract is unit-testable; AppShell only
    // wires it and delegates.
    m_shellService = std::make_unique<ShellService>(m_signalBus);
    m_shellService->setMenuParent(mainWindow);
    m_registry->registerService<IShellService>(m_shellService.get());

    // Outgoing-mail seam: UserManager (core) notifies users by email but must
    // never link the SMTP transport in services/mail directly — the adapter is
    // wired here, at the composition root.
    m_outbox = std::make_unique<SmtpOutbox>(
        m_registry->getRequired<ISmtpCredentialService>());
    UserManager::instance()->setOutgoingMail(m_outbox.get());
}

AppShell::~AppShell() {
  // UserManager is a process-lifetime static singleton that outlives this
  // shell. Detach the injected outbox here, in the destructor BODY: the body
  // runs before ANY member is destroyed, so a later UserManager email path
  // never dereferences a dangling IOutgoingMail pointer. (Do not rely on
  // m_outbox's declaration position for this — it is not the last member.)
  UserManager::instance()->setOutgoingMail(nullptr);
  // Registry không sở hữu services, nhưng phải ngừng expose chúng trước khi
  // các unique_ptr dưới đây giải phóng impl — bất kỳ get() trễ nào trong
  // teardown sẽ nhận nullptr thay vì dangling pointer.
  m_registry->clear();
}

const ServiceRegistry* AppShell::services() {
  return m_registry.get();
}

void AppShell::shutdownServices() {
  m_registry->clear();
}

// ─── initializeServices ───────────────────────────────────────────────────────
void AppShell::initializeServices() {
    // ── Phase 1: Critical services (needed for UI immediately) ────────────────
    m_sceneService    = std::make_unique<SceneService>(
        m_registry->getRequired<IUserPreferences>(), m_vtkWidget, this);
    m_viewerService = std::make_unique<ViewerService>(m_sceneService.get(), m_signalBus, this);
    m_settingsService = std::make_unique<SettingsService>(
        AppConfig::bootstrap().configPath());

    m_registry->registerService<ISceneService>(m_sceneService.get());
    m_registry->registerService<IViewerService>(m_viewerService.get());
    m_registry->registerService<ISettingsService>(m_settingsService.get());
    // IUserManager + ILanguageService + IShellService were registered in the
    // constructor (widget-independent; UI setup translates before this runs).
}

void AppShell::initializeDeferredServices() {
    // Phase-3: deferred (heavy) service wiring. Kept as one explicit block
    // here rather than a builder hierarchy — the composition root must stay
    // the single place that knows the concrete services.
    if (m_reconService) return;

    // Reconstruction module
    m_reconService = std::make_unique<ReconstructionService>();
    m_registry->registerService<IReconstructionService>(m_reconService.get());

    // AI module (builder-ready: can split into AIServiceBuilder + AIAssistantBuilder)
    m_aiService    = std::make_unique<AIService>(this);
    m_registry->registerService<IAIService>(m_aiService.get());

    // Async model loading is owned by AppCore, not by the plugin: a plugin
    // can be unloaded while the worker is still running (see
    // IAIModelLoadService). It is destroyed before m_aiService.
    m_aiModelLoadService = std::make_unique<AIModelLoadService>(m_aiService.get(), this);
    m_registry->registerService<IAIModelLoadService>(m_aiModelLoadService.get());

    // The prediction log is owned by AppCore for the same reason: the image
    // is produced by the plugin, but the worker that writes it must not be
    // plugin code on a pool nobody joins.
    m_aiPredictionLogService = std::make_unique<AIPredictionLogService>(
        m_registry->getRequired<IAppPaths>(), this);
    m_registry->registerService<IAIPredictionLogService>(m_aiPredictionLogService.get());

    // Assistant module (session state lives in AIAssistant; persistence in
    // ChatSessionStore — see src/services/ai)
    m_aiAssistant  = std::make_unique<AIAssistant>(
        m_registry->getRequired<ILanguageService>(),
        m_registry->getRequired<IAuthService>(),
        m_registry->getRequired<IUserPreferences>(),
        m_registry->getRequired<IAppPaths>(),
        AppConfig::bootstrap().pythonInterpreter(),
        AppConfig::bootstrap().aiServerPort(),
        this);
    m_registry->registerService<IAIAssistantService>(m_aiAssistant.get());

    // Mail module: three narrow contracts, not the aggregate — the mail stack
    // needs "who am I", "my profile" and "my preferences", nothing else.
    m_mailService  = std::make_unique<MailService>(
        m_registry->getRequired<IAuthService>(),
        m_registry->getRequired<IUserProfileService>(),
        m_registry->getRequired<IUserPreferences>());
    m_registry->registerService<IMailService>(m_mailService.get());
}

// ─── IAppContext ──────────────────────────────────────────────────────────────
ISceneService* AppShell::scene() {
    return m_sceneService.get();
}

IViewerService* AppShell::viewer() {
    return m_viewerService.get();
}

ISettingsService* AppShell::settings() {
    return m_settingsService.get();
}

// ─── Shell wiring ────────────────────────────────────────────────────────────
IShellService* AppShell::shell() {
    return m_shellService.get();
}

void AppShell::setMenuBar(QMenuBar* bar) {
    m_shellService->setMenuBar(bar);
}

void AppShell::setProgressDialog(IProgressDialog* dlg) {
    m_shellService->setProgressDialog(dlg);
}

// ─── Language ────────────────────────────────────────────────────────────────
QString AppShell::translate(const QString& key) {
    auto* lang = m_registry->getRequired<ILanguageService>();
    return lang ? lang->translate(key) : key;
}

void AppShell::setLanguage(const QString& lang) {
    auto* service = m_registry->getRequired<ILanguageService>();
    if (service) service->setLanguage(lang);
}
