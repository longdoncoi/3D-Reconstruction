#ifdef _WIN32
#include <windows.h>
#endif
#include <QApplication>
#include <QSurfaceFormat>
#include <QIcon>
#include <QDir>
#include <QFileInfo>
#include <vtkOpenGLRenderWindow.h>
#include "MainWindow.h"
#include "Logger.h"
#include "AppConfig.h"
#include "StyleManager.h"

int main(int argc, char* argv[])
{
    // ── Fix Qt platform plugin discovery ────────────────────────────────
    // Qt Creator + vcpkg can leave QT_PLUGIN_PATH empty, so Qt cannot
    // find platforms/qwindows.dll.  Fix: resolve the exe directory with
    // the Win32 API (always absolute) and tell Qt to search there BEFORE
    // the QApplication constructor tries to load the platform plugin.
#ifdef Q_OS_WIN
    {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        QString exeDir = QFileInfo(QString::fromWCharArray(exePath)).absolutePath();
        QApplication::addLibraryPath(exeDir);          // static – works pre-QApp
        qputenv("QT_PLUGIN_PATH", exeDir.toLocal8Bit()); // belt-and-suspenders
    }
#endif

    QApplication app(argc, argv);
    app.setWindowIcon(QIcon("f:/PROJECTS/QT/3D-Reconstruction/src/app/app_icon.png"));
    StyleManager::applyTheme(&app);
    
    // Initialize AppConfig
    AppConfig::instance().initialize(QApplication::applicationDirPath());
    
    Logger::initialize();
    
    // Tắt multisampling để tránh xung đột với Qt
    vtkOpenGLRenderWindow::SetGlobalMaximumNumberOfMultiSamples(0);

    QSurfaceFormat::setDefaultFormat(QSurfaceFormat::defaultFormat());
    MainWindow window;
    window.show();

    // Force the window to paint immediately — deferred services and plugins
    // will initialize on the next event loop iteration (via QTimer::singleShot).
    QApplication::processEvents();

    return app.exec();
}
