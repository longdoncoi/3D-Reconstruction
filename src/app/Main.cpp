#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
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
    // Set window icon — prefer embedded Qt resource, fall back to file on disk
    {
        QIcon appIcon(QStringLiteral(":/app_icon.png"));
        if (appIcon.isNull()) {
            // Fallback: icon beside executable (production portable)
            QString iconPath = QApplication::applicationDirPath() + QStringLiteral("/app_icon.png");
            if (!QFileInfo::exists(iconPath)) {
                // Fallback for dev builds where icon lives in source tree
                iconPath = QDir::cleanPath(QApplication::applicationDirPath() + QStringLiteral("/../../src/app/app_icon.png"));
            }
            if (QFileInfo::exists(iconPath)) {
                appIcon = QIcon(iconPath);
            }
        }
        if (!appIcon.isNull()) {
            app.setWindowIcon(appIcon);
        }
    }
    StyleManager::applyTheme(&app);
    
    // Initialize AppConfig
    AppConfig::instance().initialize(QApplication::applicationDirPath());
    
    // ── Fix UTF-8 output cho Qt Creator Application Output ────────────────
    // stdout mặc định trên Windows dùng hệ thống codepage (CP1252/CP1258)
    // gây ra ký tự tiếng Việt bị mã hóa sai khi Qt Creator đọc.
    // _setmode(_O_BINARY) + SetConsoleOutputCP(CP_UTF8) đảm bảo bytes UTF-8
    // thuần túy được gửi ra stdout mà không bị Windows API chuyển đổi.
#ifdef Q_OS_WIN
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif

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
