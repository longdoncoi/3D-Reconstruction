#include "AppConfig.h"
#include <QCoreApplication>
#include <QtTest>

class TestAppConfig : public QObject {
  Q_OBJECT

private slots:
  void initTestCase() {
    // AppConfig::appDir() only reflects whatever was passed to initialize();
    // production code calls this from main() before anything reads paths,
    // so the test needs to do the same before asserting on them.
    AppConfig::instance().initialize(QCoreApplication::applicationDirPath());
  }

  void testSingletonInstance() {
    AppConfig &instance1 = AppConfig::instance();
    AppConfig &instance2 = AppConfig::instance();
    QCOMPARE(&instance1, &instance2);
  }

  void testDirectoryPaths() {
    AppConfig &config = AppConfig::instance();

    // This will verify that the paths are not empty
    QVERIFY(!config.appDir().isEmpty());
    QVERIFY(!config.configPath().isEmpty());
    QVERIFY(!config.logsDir().isEmpty());
    QVERIFY(!config.pluginsDir().isEmpty());
  }
};

QTEST_GUILESS_MAIN(TestAppConfig)

#include "TestAppConfig.moc"