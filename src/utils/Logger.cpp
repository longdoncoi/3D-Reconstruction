#include "Logger.h"
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <iostream>
#include <mutex>
#include "AppConfig.h"

namespace Logger {

static std::mutex s_logMutex;

QString getLogFilePath() {
  QString logsDir = AppConfig::instance().logsDir();
  QDir().mkpath(logsDir);
  QString currentDate = QDateTime::currentDateTime().toString("yyyy-MM-dd");
  return logsDir + "/" + currentDate + ".txt";
}

void customMessageHandler(QtMsgType type, const QMessageLogContext &context,
                          const QString &msg) {
  QString txt;
  QString timestamp =
      QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");

  switch (type) {
  case QtDebugMsg:
    txt = QString("[%1] [DEBUG] %2").arg(timestamp).arg(msg);
    break;
  case QtInfoMsg:
    txt = QString("[%1] [INFO] %2").arg(timestamp).arg(msg);
    break;
  case QtWarningMsg:
    txt = QString("[%1] [WARNING] %2").arg(timestamp).arg(msg);
    break;
  case QtCriticalMsg:
    txt = QString("[%1] [CRITICAL] %2").arg(timestamp).arg(msg);
    break;
  case QtFatalMsg:
    txt = QString("[%1] [FATAL] %2").arg(timestamp).arg(msg);
    break;
  }

  std::lock_guard<std::mutex> lock(s_logMutex);

  // Print to terminal
  std::cout << txt.toLocal8Bit().constData() << std::endl;

  // Write to log file
  QFile outFile(getLogFilePath());
  if (outFile.open(QIODevice::WriteOnly | QIODevice::Append)) {
    QTextStream ts(&outFile);
    // ts.setCodec("UTF-8"); // Optional in Qt5, defaults to UTF-8 in Qt6
    ts << txt << "\n";
    outFile.close();
  }

  if (type == QtFatalMsg) {
    abort();
  }
}

void initialize() { qInstallMessageHandler(customMessageHandler); }
} // namespace Logger
