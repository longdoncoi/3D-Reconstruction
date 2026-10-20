#include <QtTest>
#include <QApplication>
#include <QDir>
#include <QMainWindow>
#include <QPointer>
#include <QStandardPaths>

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkSmartPointer.h>

#include "AppShell.h"
#include "AppConfig.h"
#include "IAIAssistantService.h"
#include "IAIModelLoadService.h"
#include "IAIService.h"
#include "IAppPaths.h"
#include "ILanguageService.h"
#include "ILicenseService.h"
#include "IMailService.h"
#include "IReconstructionService.h"
#include "ISceneService.h"
#include "ISettingsService.h"
#include "IShellService.h"
#include "IUserManager.h"
#include "IUserPreferences.h"
#include "IViewerService.h"
#include "SignalBus.h"

/**
 * @brief Composition-root wiring test for AppShell.
 *
 * AppShell is the one component every plugin trusts through IAppContext, and
 * no unit test covered it: a service registered in the wrong phase, a missing
 * registration or a getter returning a different instance than the registry
 * stays invisible until a plugin dereferences a null/stale pointer at
 * runtime. This test pins the wiring contract:
 *
 *   1. initializeServices() registers the phase-1 surface (widget-independent
 *      kernel services + UI-critical services) and nothing deferred.
 *   2. initializeDeferredServices() registers reconstruction/AI/mail exactly
 *      once (idempotent) without replacing existing instances.
 *   3. The IAppContext getters are the same objects as the registry entries —
 *      two access paths, one instance.
 *   4. Teardown runs cleanly and a second composition root can be built on
 *      top of the process singletons (no hidden static state).
 *
 * Runs offscreen (Ctest sets QT_QPA_PLATFORM=offscreen): the QT VTK widget is
 * constructed and given a render window but never shown, so no OpenGL context
 * is created — SceneService only needs a render window object to attach its
 * renderer at construction time.
 */
class TestAppShell : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void phaseOneRegistersCoreServicesOnly();
    void contextGettersAreTheRegistryInstances();
    void deferredPhaseIsIdempotent();
    void teardownIsCleanAndRecomposable();

private:
    QMainWindow m_mainWindow;
    QVTKOpenGLNativeWidget m_vtkWidget;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> m_renderWindow;
    std::unique_ptr<AppShell> m_shell;
    ServiceRegistry* m_registry = nullptr;
    QString m_missingKey;
};

void TestAppShell::initTestCase()
{
    // Sandbox every path a service may write (chat_sessions_*.json, Logs/, …):
    // a sandbox directory without a CMakeLists.txt parent puts AppConfig in
    // "production" mode, so dataRoot resolves under the test-mode AppData
    // location instead of the repository tree.
    QStandardPaths::setTestModeEnabled(true);
    const QString sandbox = QDir::tempPath() + QStringLiteral("/3d_recon_appshell");
    QDir().mkpath(sandbox);
    QT_WARNING_PUSH
    QT_WARNING_DISABLE_DEPRECATED
    AppConfig::bootstrap().initialize(sandbox);
    QT_WARNING_POP

    m_missingKey = QStringLiteral("test.key.that.does.not.exist");

    // The widget is never shown; an explicit render window keeps the test
    // independent of QVTKOpenGLNativeWidget's lazy construction.
    m_renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    m_vtkWidget.setRenderWindow(m_renderWindow.Get());

    m_shell = std::make_unique<AppShell>(&m_mainWindow, &m_vtkWidget, nullptr);
    m_registry = m_shell->services();
    QVERIFY(m_registry);
    m_shell->initializeServices();
}

void TestAppShell::phaseOneRegistersCoreServicesOnly()
{
    // Widget-independent kernel services registered at construction time
    // (translate() runs before initializeServices() in production).
    QVERIFY(m_registry->get<IUserManager>());
    QVERIFY(m_registry->get<IUserPreferences>());
    QVERIFY(m_registry->get<ILicenseService>());
    QVERIFY(m_registry->get<ILanguageService>());
    QVERIFY(m_registry->get<IAppPaths>());
    QVERIFY(m_registry->get<IShellService>());

    // UI-critical phase-1 services.
    QVERIFY(m_registry->get<ISceneService>());
    QVERIFY(m_registry->get<IViewerService>());
    QVERIFY(m_registry->get<ISettingsService>());

    // Deferred services must stay unregistered until MainWindow asks for
    // them right before loadPlugins().
    QVERIFY(!m_registry->get<IReconstructionService>());
    QVERIFY(!m_registry->get<IAIService>());
    QVERIFY(!m_registry->get<IAIModelLoadService>());
    QVERIFY(!m_registry->get<IAIAssistantService>());
    QVERIFY(!m_registry->get<IMailService>());
}

void TestAppShell::contextGettersAreTheRegistryInstances()
{
    // Two-tier contract: the IAppContext getters and the ServiceRegistry must
    // hand out the same instance, or a consumer resolving through one path
    // would observe a different object than a consumer using the other.
    QCOMPARE(m_shell->scene(), m_registry->get<ISceneService>());
    QCOMPARE(m_shell->viewer(), m_registry->get<IViewerService>());
    QCOMPARE(m_shell->settings(), m_registry->get<ISettingsService>());
    QCOMPARE(m_shell->shell(), m_registry->get<IShellService>());
    QCOMPARE(m_shell->services(), m_registry);
    QVERIFY(m_shell->signalBus());

    // translate() must degrade to the key (never an empty string) when a
    // service or a dictionary entry is missing.
    QCOMPARE(m_shell->translate(m_missingKey), m_missingKey);
    m_shell->setLanguage(QStringLiteral("en"));
}

void TestAppShell::deferredPhaseIsIdempotent()
{
    m_shell->initializeDeferredServices();

    ISceneService* scene = m_registry->get<ISceneService>();
    auto* recon = m_registry->get<IReconstructionService>();
    auto* ai = m_registry->get<IAIService>();
    auto* modelLoader = m_registry->get<IAIModelLoadService>();
    auto* assistant = m_registry->get<IAIAssistantService>();
    auto* mail = m_registry->get<IMailService>();
    QVERIFY(scene);
    QVERIFY(recon);
    QVERIFY(ai);
    QVERIFY(modelLoader);
    QVERIFY(assistant);
    QVERIFY(mail);

    // MainWindow may call this more than once; live instances held by dock
    // widgets and plugins must not be silently replaced.
    m_shell->initializeDeferredServices();
    QCOMPARE(m_registry->get<ISceneService>(), scene);
    QCOMPARE(m_registry->get<IReconstructionService>(), recon);
    QCOMPARE(m_registry->get<IAIService>(), ai);
    QCOMPARE(m_registry->get<IAIModelLoadService>(), modelLoader);
    QCOMPARE(m_registry->get<IAIAssistantService>(), assistant);
    QCOMPARE(m_registry->get<IMailService>(), mail);
}

void TestAppShell::teardownIsCleanAndRecomposable()
{
    QPointer<SignalBus> bus = m_shell->signalBus();
    m_shell.reset();
    QVERIFY(bus.isNull()); // SignalBus is parented to the shell and dies with it.

    // A second composition root must build on top of the process singletons
    // (UserManager/AppConfig/LanguageManager) without stale static state.
    m_shell = std::make_unique<AppShell>(&m_mainWindow, &m_vtkWidget, nullptr);
    m_registry = m_shell->services();
    QVERIFY(m_registry);
    m_shell->initializeServices();
    m_shell->initializeDeferredServices();
    QVERIFY(m_registry->get<IMailService>());
    QVERIFY(m_registry->get<ISceneService>());
    m_shell.reset();
}

QTEST_MAIN(TestAppShell)

#include "TestAppShell.moc"
