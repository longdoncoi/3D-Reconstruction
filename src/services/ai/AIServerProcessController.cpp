#include "AIServerProcessController.h"

#include "AppConstants.h"
#include "IAppPaths.h"
#include "ILanguageService.h"

#include <QDebug>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QStandardPaths>

#include <utility>

AIServerProcessController::AIServerProcessController(ILanguageService *lang,
                                                     IAppPaths *paths,
                                                     QString pythonInterpreter,
                                                     QObject *parent)
    : QObject(parent)
    , m_lang(lang)
    , m_paths(paths)
    , m_pythonInterpreter(pythonInterpreter.isEmpty() ? QStringLiteral("python")
                                                      : std::move(pythonInterpreter))
    , m_process(new QProcess(this))
{
    connectProcess();
}

AIServerProcessController::~AIServerProcessController()
{
    stop();
}

void AIServerProcessController::connectProcess()
{
    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &AIServerProcessController::onReadyRead);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &AIServerProcessController::onErrorRead);
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &AIServerProcessController::onFinished);
}

void AIServerProcessController::disconnectProcess()
{
    disconnect(m_process, nullptr, this, nullptr);
}

void AIServerProcessController::start(int modelIndex)
{
    const QStringList models = AppConstants::AIAssistant::modelNames();
    const QString modelName =
        (modelIndex >= 0 && modelIndex < models.size()) ? models.at(modelIndex)
                                                        : QStringLiteral("Unknown Model");
    qDebug() << "[AI Server] Starting server with model index:" << modelIndex
             << "(" << modelName << ")";

    m_readyEmitted = false;
    disconnectProcess();
    connectProcess();

    const QString script = (m_paths ? m_paths->aiAssistantDir() : QString())
                           + "/" + AppConstants::AIServer::chatbotScript();
    if (!QFileInfo::exists(script)) {
        if (m_lang) emit errorOccurred(m_lang->translate("ai.missing_script"));
        return;
    }

    if (m_lang) emit serverStatusChanged(m_lang->translate("ai.starting_server"));

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("PYTHONUNBUFFERED", "1");
    env.insert("PYTHONIOENCODING", "utf-8");
    env.insert("APP_DATA_DIR", QStandardPaths::writableLocation(QStandardPaths::AppDataLocation));
    m_process->setProcessEnvironment(env);
    m_process->setWorkingDirectory(m_paths->aiAssistantDir());
    m_process->start(m_pythonInterpreter,
                     QStringList() << "-u" << script << QString::number(modelIndex));
}

void AIServerProcessController::stop()
{
    disconnectProcess();

    if (m_process->state() == QProcess::NotRunning) return;

    // Keep the detached AI-server process running after Qt exits: terminate
    // the wrapper first, then kill it only when the bounded wait expires.
    m_process->terminate();
    if (!m_process->waitForFinished(1000)) {
        m_process->kill();
        m_process->waitForFinished(AppConstants::AIServer::STOP_SERVER_TIMEOUT_MS);
    }
}

bool AIServerProcessController::isRunning() const
{
    return m_process && m_process->state() != QProcess::NotRunning;
}

void AIServerProcessController::detectReady(const QString &output)
{
    if (m_readyEmitted) return;
    if (output.contains("[SUCCESS] AI Server") ||
        output.contains("Uvicorn running on") ||
        output.contains("Application startup complete")) {
        m_readyEmitted = true;
        emit serverReady();
    }
}

void AIServerProcessController::onReadyRead()
{
    const QString output = QString::fromUtf8(m_process->readAllStandardOutput());
    if (!output.isEmpty())
        qDebug().noquote() << QString("[AIAssistant Server]\n%1").arg(output.trimmed());

    const QStringList lines = output.split('\n');
    for (const QString &line : lines) {
        const int index = line.indexOf("[STATUS]");
        if (index >= 0 && m_lang)
            emit serverStatusChanged(line.mid(index + 8).trimmed());
    }

    detectReady(output);
}

void AIServerProcessController::onErrorRead()
{
    const QString output = QString::fromUtf8(m_process->readAllStandardError());
    if (!output.isEmpty())
        qDebug().noquote() << QString("[AIAssistant Server]\n%1").arg(output.trimmed());

    const QStringList lines = output.split('\n');
    for (const QString &line : lines) {
        const int index = line.indexOf("[STATUS]");
        if (index >= 0 && m_lang)
            emit serverStatusChanged(line.mid(index + 8).trimmed());
    }

    detectReady(output);
}

void AIServerProcessController::onFinished(int exitCode, QProcess::ExitStatus status)
{
    m_readyEmitted = false;
    emit processFinished(exitCode, status == QProcess::CrashExit);
}
