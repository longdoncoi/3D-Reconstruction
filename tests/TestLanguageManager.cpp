#include "LanguageManager.h"
#include <QtTest>


class TestLanguageManager : public QObject {
  Q_OBJECT

private slots:
  void testSingletonInstance() {
    LanguageManager &lm1 = LanguageManager::instance();
    LanguageManager &lm2 = LanguageManager::instance();
    QCOMPARE(&lm1, &lm2);
  }

  void testSetLanguage() {
    LanguageManager &lm = LanguageManager::instance();

    QSignalSpy spy(&lm, &LanguageManager::languageChanged);

    lm.setLanguage("en");
    QCOMPARE(lm.currentLanguage(), QString("en"));
    QCOMPARE(spy.count(), 1);

    lm.setLanguage("vi");
    QCOMPARE(lm.currentLanguage(), QString("vi"));
    QCOMPARE(spy.count(), 2);
  }

  void testTranslateFallback() {
    LanguageManager &lm = LanguageManager::instance();

    // When key doesn't exist, translate should fallback to key
    QString fallback = lm.translate("non.existent.key");
    QCOMPARE(fallback, QString("non.existent.key"));
  }
};

QTEST_GUILESS_MAIN(TestLanguageManager)

#include "TestLanguageManager.moc"