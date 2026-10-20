#include "AppConfig.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSettings>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QtTest>

class TestAppConfig : public QObject {
  Q_OBJECT

private slots:
  void initTestCase()
  {
    // Keep every write inside the test sandbox: the production-layout branch
    // resolves dataRoot through QStandardPaths.
    QStandardPaths::setTestModeEnabled(true);
    AppConfig::bootstrap().initialize(QCoreApplication::applicationDirPath());
  }

  void cleanupTestCase()
  {
    QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation)
         + QStringLiteral("/3D-Reconstruction")).removeRecursively();
    QStandardPaths::setTestModeEnabled(false);
  }

  void init()
  {
    // Each slot starts from the development layout (the build tree reaches
    // the repository root and therefore Config/ in-tree).
    AppConfig::bootstrap().initialize(QCoreApplication::applicationDirPath());
  }

  void testSingletonInstance()
  {
    AppConfig &instance1 = AppConfig::bootstrap();
    AppConfig &instance2 = AppConfig::bootstrap();
    QCOMPARE(&instance1, &instance2);
  }

  void testDirectoryPaths()
  {
    AppConfig &config = AppConfig::bootstrap();

    QVERIFY(!config.appDir().isEmpty());
    QVERIFY(!config.projectRootDir().isEmpty());
    QVERIFY(!config.dataRootDir().isEmpty());
    QVERIFY(!config.configPath().isEmpty());
    QVERIFY(!config.logsDir().isEmpty());
    QVERIFY(!config.thumbnailsDir().isEmpty());
    QVERIFY(!config.pluginsDir().isEmpty());
  }

  void testPathMatrixUsesDocumentedRoots()
  {
    AppConfig &config = AppConfig::bootstrap();
    const QString dataRoot = config.dataRootDir();
    const QString projectRoot = config.projectRootDir();

    QCOMPARE(config.configPath(),
             QDir::cleanPath(dataRoot + QStringLiteral("/Config/Config.ini")));
    QCOMPARE(config.logsDir(), QDir::cleanPath(dataRoot + QStringLiteral("/Logs")));
    QCOMPARE(config.predictDir(QStringLiteral("mesh")),
             QDir::cleanPath(dataRoot + QStringLiteral("/Predict/mesh")));
    QCOMPARE(config.uploadDir(), QDir::cleanPath(dataRoot + QStringLiteral("/Upload")));
    QCOMPARE(config.thumbnailsDir(),
             QDir::cleanPath(dataRoot + QStringLiteral("/Thumbnails")));
    QCOMPARE(config.modelsDir(),
             QDir::cleanPath(projectRoot + QStringLiteral("/AIComputerVision/Models")));
    QCOMPARE(config.aiAssistantDir(),
             QDir::cleanPath(projectRoot + QStringLiteral("/AIAssistant")));
    QCOMPARE(config.aiComputerVisionDir(),
             QDir::cleanPath(projectRoot + QStringLiteral("/AIComputerVision")));
    QCOMPARE(config.pluginsDir(),
             QDir::cleanPath(config.appDir() + QStringLiteral("/plugins")));
  }

  void testDevelopmentLayoutFindsProjectRoot()
  {
    QTemporaryDir repo;
    QVERIFY(repo.isValid());

    QFile marker(repo.filePath(QStringLiteral("CMakeLists.txt")));
    QVERIFY(marker.open(QIODevice::WriteOnly));
    marker.write("project(marker)\n");
    marker.close();

    const QString appDir = repo.filePath(QStringLiteral("build/Release"));
    QVERIFY(QDir().mkpath(appDir));

    AppConfig::bootstrap().initialize(appDir);
    AppConfig &config = AppConfig::bootstrap();

    QCOMPARE(config.appDir(), QDir(appDir).absolutePath());
    QCOMPARE(config.projectRootDir(), QDir(repo.path()).absolutePath());
    QCOMPARE(config.dataRootDir(), config.projectRootDir());
    QVERIFY(QFileInfo(config.configPath()).absoluteDir().exists());
  }

  void testProductionLayoutUsesWritableAppData()
  {
    QTemporaryDir install;
    QVERIFY(install.isValid());

    AppConfig::bootstrap().initialize(install.path());
    AppConfig &config = AppConfig::bootstrap();

    QCOMPARE(config.projectRootDir(), QDir(install.path()).absolutePath());
    QVERIFY(config.dataRootDir() != config.projectRootDir());

    const QString appData =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QCOMPARE(config.dataRootDir(),
             QDir::cleanPath(appData + QStringLiteral("/3D-Reconstruction")));
    QVERIFY(QFileInfo(config.configPath()).absoluteDir().exists());
  }

  void testAiSettingsAreValidated()
  {
    QTemporaryDir install;
    QVERIFY(install.isValid());
    AppConfig::bootstrap().initialize(install.path());
    AppConfig &config = AppConfig::bootstrap();

    QSettings settings(config.configPath(), QSettings::IniFormat);
    settings.setValue(QStringLiteral("AI/python_interpreter"),
                      QStringLiteral("  C:/Python311/python.exe  "));
    settings.setValue(QStringLiteral("AI/server_port"), 1234);
    settings.sync();
    QCOMPARE(config.pythonInterpreter(), QStringLiteral("C:/Python311/python.exe"));
    QCOMPARE(config.aiServerPort(), 1234);

    // Blank interpreter and an out-of-range port fall back to the defaults
    // instead of launching an empty command / binding an invalid port.
    settings.setValue(QStringLiteral("AI/python_interpreter"), QStringLiteral("   "));
    settings.setValue(QStringLiteral("AI/server_port"), 65536);
    settings.sync();
    QCOMPARE(config.pythonInterpreter(), QStringLiteral("python"));
    QCOMPARE(config.aiServerPort(), 8080);

    settings.setValue(QStringLiteral("AI/server_port"), -1);
    settings.sync();
    QCOMPARE(config.aiServerPort(), 8080);

    // Missing keys use the same defaults.
    settings.remove(QStringLiteral("AI"));
    settings.sync();
    QCOMPARE(config.pythonInterpreter(), QStringLiteral("python"));
    QCOMPARE(config.aiServerPort(), 8080);
  }
};

QTEST_GUILESS_MAIN(TestAppConfig)

#include "TestAppConfig.moc"
