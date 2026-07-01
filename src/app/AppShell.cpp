#include "AppShell.h"
#include "ISceneService.h"
#include "IViewerService.h"
#include "ISettingsService.h"
#include "SceneService.h"
#include "ViewerService.h"
#include "SettingsService.h"
#include "ReconstructionService.h"
#include "AIService.h"
#include "IReconstructionService.h"
#include "IAIService.h"
#include "IAIAssistantService.h"
#include "IMailService.h"
#include "MailService.h"
#include "LanguageManager.h"
#include "CustomProgressDialog.h"
#include "ReconstructionPipeline.h"
#include "AIProcessor.h"
#include "AIAssistant.h"
#include <QMenuBar>
#include <QMenu>
#include <QWidget>
#include <QVTKOpenGLNativeWidget.h>

AppShell::AppShell(QMainWindow*           mainWindow,
                   QVTKOpenGLNativeWidget* vtkWidget,
                   QObject*               parent)
    : QObject(parent)
    , m_mainWindow(mainWindow)
    , m_vtkWidget(vtkWidget)
    , m_registry(std::make_unique<ServiceRegistry>())
{
    m_signalBus = new SignalBus(this);
}

AppShell::~AppShell() = default;

// ─── initializeServices ───────────────────────────────────────────────────────
void AppShell::initializeServices() {
    // 1. Construct services
    m_sceneService    = std::make_unique<SceneService>(this, m_vtkWidget, this);
    m_viewerService   = std::make_unique<ViewerService>(this, this);
    m_settingsService = std::make_unique<SettingsService>();
    m_reconService    = std::make_unique<ReconstructionService>();
    m_aiService       = std::make_unique<AIService>(this);
    m_aiAssistant     = std::make_unique<AIAssistant>(this);
    m_mailService     = std::make_unique<MailService>();

    // 2. Register vào ServiceRegistry (plugin lookup bằng interface type)
    m_registry->registerService<ISceneService>(m_sceneService.get());
    m_registry->registerService<IViewerService>(m_viewerService.get());
    m_registry->registerService<ISettingsService>(m_settingsService.get());
    m_registry->registerService<IReconstructionService>(m_reconService.get());
    m_registry->registerService<IAIService>(m_aiService.get());
    m_registry->registerService<IAIAssistantService>(m_aiAssistant.get());
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

// ─── Menu ────────────────────────────────────────────────────────────────────
QMenu* AppShell::getMenu(const QString& id) {
    if (!m_menus.contains(id)) {
        m_menus[id] = new QMenu(m_mainWindow);
    }
    return m_menus[id];
}

QMenuBar* AppShell::menuBar() {
    return m_menuBar;
}

void AppShell::updateMenuStates() {
    emit m_signalBus->stateChanged();
}

// ─── Tab Panels ──────────────────────────────────────────────────────────────
void AppShell::registerTabPanel(const QString& key, QWidget* panel) {
    m_tabPanels[key] = panel;
}

QWidget* AppShell::getTabPanel(const QString& tabName) {
    // Direct key lookup
    if (m_tabPanels.contains(tabName)) return m_tabPanels[tabName];
    // Fallback: search by translated label
    for (auto it = m_tabPanels.begin(); it != m_tabPanels.end(); ++it) {
        if (LM_TR(it.key()) == tabName) return it.value();
    }
    return nullptr;
}

// ─── Language ────────────────────────────────────────────────────────────────
QString AppShell::translate(const QString& key) {
    return LM_TR(key);
}

void AppShell::setLanguage(const QString& lang) {
    LM.setLanguage(lang);
}
