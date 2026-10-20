#ifndef AI_SERVER_PROCESS_CONTROLLER_H
#define AI_SERVER_PROCESS_CONTROLLER_H

#include <QObject>
#include <QProcess>
#include <QString>

#include "Global.h"

class IAppPaths;
class ILanguageService;

/**
 * @brief Owns the Python AI-server process and its readiness handshake.
 *
 * AIAssistant keeps the request/session state machine; this controller owns
 * only process lifecycle (start/stop/terminate), environment setup and the
 * "[SUCCESS] AI Server" / "Uvicorn running" readiness detection. Keeping the
 * QProcess out of the orchestrator makes restart/shutdown paths testable and
 * prevents blocking waits from leaking into request code.
 */
class APP_EXPORT AIServerProcessController : public QObject {
    Q_OBJECT
public:
    AIServerProcessController(ILanguageService *lang, IAppPaths *paths,
                              QString pythonInterpreter, QObject *parent = nullptr);
    ~AIServerProcessController() override;

    void start(int modelIndex);
    void stop();
    bool isRunning() const;

signals:
    void serverStatusChanged(const QString &status);
    void errorOccurred(const QString &message);
    void serverReady();
    void processFinished(int exitCode, bool crashed);

private slots:
    void onReadyRead();
    void onErrorRead();
    void onFinished(int exitCode, QProcess::ExitStatus status);

private:
    void connectProcess();
    void disconnectProcess();
    void detectReady(const QString &output);

    ILanguageService *m_lang = nullptr;
    IAppPaths *m_paths = nullptr;
    QString m_pythonInterpreter;
    QProcess *m_process = nullptr;
    bool m_readyEmitted = false;
};

#endif // AI_SERVER_PROCESS_CONTROLLER_H
