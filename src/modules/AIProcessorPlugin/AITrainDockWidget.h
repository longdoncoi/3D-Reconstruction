#ifndef AITRAINDOCKWIDGET_H
#define AITRAINDOCKWIDGET_H

#include <QDockWidget>
#include <QProcess>
#include <QTextBrowser>
#include <QProgressBar>
#include <QPushButton>

class IAppContext;

class AITrainDockWidget : public QDockWidget {
    Q_OBJECT
public:
    explicit AITrainDockWidget(IAppContext* ctx, QWidget* parent = nullptr);
    ~AITrainDockWidget();

    void startTraining(bool autoConfirm = false);

private slots:
    void closeTrainDock();
    void onStopTrain();
    void onTrainProcessOutput();
    void onTrainProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    IAppContext* m_ctx;
    QProcess* trainProcess;
    QProgressBar* trainProgressBar;
    QTextBrowser* trainLogBrowser;
    QPushButton* btnStopTrain;
};

#endif // AITRAINDOCKWIDGET_H
