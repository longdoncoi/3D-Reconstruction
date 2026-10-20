#include "AITrainDockWidget.h"
#include "IAppContext.h"
#include "IAppPaths.h"
#include "IShellService.h"
#include "AppConstants.h"
#include "../../utils/ModernMessageBox.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QScrollBar>
#include <QDialog>
#include <QCheckBox>
#include <QDebug>
#include <QDialogButtonBox>
#include <QTimer>

AITrainDockWidget::AITrainDockWidget(IAppContext* ctx, QWidget* parent)
    : QDockWidget(ctx->translate("aiproc.train_title"), parent), m_ctx(ctx) 
{
    setMinimumWidth(520);
    QWidget     *dw = new QWidget(this);
    QVBoxLayout *dl = new QVBoxLayout(dw);
    dl->setContentsMargins(0, 0, 0, 0);
    dl->setSpacing(0);

    QWidget     *topBar = new QWidget(dw);
    QHBoxLayout *tl     = new QHBoxLayout(topBar);
    tl->setContentsMargins(8, 6, 8, 6);
    tl->setSpacing(6);
    topBar->setStyleSheet("background:#1a1a1f; border-bottom:1px solid #2e2e3a;");
    
    m_titleLabel = new QLabel(m_ctx->translate("aiproc.train_title"), topBar);
    m_titleLabel->setStyleSheet("color:#ddd; font-weight:bold; font-size:14px;");

    trainProgressBar = new QProgressBar(topBar);
    trainProgressBar->setTextVisible(false);
    trainProgressBar->setStyleSheet(
        "QProgressBar { background:#2a2a35; border:1px solid #3a3a4a; border-radius:6px; color:#fff; text-align:center; height: 24px; }"
        "QProgressBar::chunk { background-color: #10a37f; border-radius:5px; }"
    );
    trainProgressBar->setRange(0, 0);
    trainProgressBar->hide();
    
    btnStopTrain = new QPushButton(m_ctx->translate("aiproc.stop"), topBar);
    btnStopTrain->setStyleSheet(
        "QPushButton { background:#d32f2f; color:#fff; border:none; border-radius:6px; padding:4px 12px; font-weight:bold; height: 24px; }"
        "QPushButton:hover { background:#b71c1c; }"
        "QPushButton:disabled { background:#555; color:#888; }"
    );
    btnStopTrain->hide();

    QPushButton *btnCloseTrain = new QPushButton("X", topBar);
    btnCloseTrain->setFixedSize(30, 30);
    btnCloseTrain->setStyleSheet(
        "QPushButton { background:transparent; color:#f1f5f9; font-size:16px; font-weight:bold; border:none; padding:0; margin:0; }"
        "QPushButton:hover { color:#ffffff; background:#ef4444; border-radius:15px; }"
    );
    
    tl->addWidget(m_titleLabel, 0);
    tl->addStretch(1);
    tl->addWidget(trainProgressBar, 2);
    tl->addWidget(btnStopTrain, 0);
    tl->addStretch(1);
    tl->addWidget(btnCloseTrain, 0);
    
    dl->addWidget(topBar);
    
    trainLogBrowser = new QTextBrowser(dw);
    trainLogBrowser->setStyleSheet("QTextBrowser { background:#1e1e24; color:#ececec; border:none; padding:12px; font-family: Consolas; font-size:13px; }");
    dl->addWidget(trainLogBrowser, 1);
    
    setWidget(dw);
    setFeatures(QDockWidget::NoDockWidgetFeatures);
    setTitleBarWidget(new QWidget());

    m_ctx->shell()->addDockWidget(Qt::RightDockWidgetArea, this);
    hide();

    trainProcess = new QProcess(this);
    
    connect(btnCloseTrain, &QPushButton::clicked, this, &AITrainDockWidget::closeTrainDock);
    connect(btnStopTrain, &QPushButton::clicked, this, &AITrainDockWidget::onStopTrain);
    connect(trainProcess, &QProcess::readyReadStandardOutput, this, &AITrainDockWidget::onTrainProcessOutput);
    connect(trainProcess, &QProcess::readyReadStandardError, this, &AITrainDockWidget::onTrainProcessOutput);
    connect(trainProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &AITrainDockWidget::onTrainProcessFinished);
    connect(trainProcess, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error != QProcess::FailedToStart) return;
        trainLogBrowser->append("<font color='#d32f2f'>" + m_ctx->translate("aiproc.python_start_failed") + "</font>");
        trainProgressBar->hide();
        btnStopTrain->hide();
    });
}

AITrainDockWidget::~AITrainDockWidget() {
    if (trainProcess->state() != QProcess::NotRunning) {
        // kill() + an explicit wait bound: the argument-less waitForFinished
        // call waits Qt's 30 s default and would freeze app teardown if the
        // child process hangs. Shutdown must stay bounded.
        trainProcess->kill();
        if (!trainProcess->waitForFinished(5000)) {
            qWarning() << "AITrainDockWidget: training process did not exit within 5 s"
                          " after kill(); continuing shutdown";
        }
    }
}

void AITrainDockWidget::retranslate() {
    setWindowTitle(m_ctx->translate("aiproc.train_title"));
    if (m_titleLabel) m_titleLabel->setText(m_ctx->translate("aiproc.train_title"));
    if (btnStopTrain) btnStopTrain->setText(m_ctx->translate("aiproc.stop"));
}

void AITrainDockWidget::startTraining(bool autoConfirm) {
    show();
    if (trainProcess->state() == QProcess::NotRunning) {
        auto* paths = m_ctx->services()->getRequired<IAppPaths>();
        if (!paths) return;
        QString scriptPath = paths->aiComputerVisionDir() + "/" + AppConstants::AIProcessor::trainScript();
        QString modelsPath = paths->modelsDir();
        const QStringList modelFiles = {
            AppConstants::AIProcessor::detectionModelFile(),
            AppConstants::AIProcessor::segmentationModelFile(),
            AppConstants::AIProcessor::trackingModelFile(),
        };
        bool modelExists = false;
        for (const QString& file : modelFiles) {
            modelExists = modelExists || QFile::exists(modelsPath + "/" + file);
        }
        
        QDialog dialog(m_ctx->shell()->dialogParent());
        dialog.setWindowTitle(m_ctx->translate("ai.training"));
        dialog.setStyleSheet("QDialog { background-color: #1a1a1f; color: white; } QCheckBox { color: white; font-size: 14px; margin: 5px; }");
        
        QVBoxLayout *layout = new QVBoxLayout(&dialog);
        QLabel* lbl = new QLabel(m_ctx->translate("aiproc.select_models"), &dialog);
        lbl->setStyleSheet("color: white; font-size: 14px; font-weight: bold;");
        layout->addWidget(lbl);
        
        QCheckBox *chkDet = new QCheckBox(m_ctx->translate("aiproc.train_model_det"), &dialog);
        QCheckBox *chkSeg = new QCheckBox(m_ctx->translate("aiproc.train_model_seg"), &dialog);
        QCheckBox *chkTrack = new QCheckBox(m_ctx->translate("aiproc.train_model_track"), &dialog);
        
        chkDet->setChecked(true);
        
        layout->addWidget(chkDet);
        layout->addWidget(chkSeg);
        layout->addWidget(chkTrack);
        
        QDialogButtonBox *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        btnBox->setStyleSheet("QPushButton { background-color: #3b82f6; color: white; padding: 5px 15px; border-radius: 4px; } QPushButton:hover { background-color: #2563eb; }");
        connect(btnBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
        connect(btnBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        layout->addWidget(btnBox);
        
        if (autoConfirm) {
            QTimer::singleShot(0, &dialog, &QDialog::accept);
        }
        if (dialog.exec() != QDialog::Accepted) {
            return;
        }

        if (!chkDet->isChecked() && !chkSeg->isChecked() && !chkTrack->isChecked()) {
            ModernMessageBox::warning(m_ctx->shell()->dialogParent(), m_ctx->translate("ai.training"),
                                      m_ctx->translate("aiproc.select_one_model"));
            return;
        }

        if (modelExists && !autoConfirm) {
            if (!ModernMessageBox::question(m_ctx->services()->getRequired<ILanguageService>(), m_ctx->shell()->dialogParent(), m_ctx->translate("aiproc.confirm"), m_ctx->translate("aiproc.model_exists_retrain"))) {
                return;
            }
        }
    
        trainLogBrowser->clear();
        trainLogBrowser->append("<font color='#10a37f'>" + m_ctx->translate("aiproc.train_started") + "</font>");
        trainProgressBar->setRange(0, 0);
        trainProgressBar->setValue(0);
        trainProgressBar->show();
        btnStopTrain->show();
        btnStopTrain->setEnabled(true);
        
        trainProcess->setWorkingDirectory(QFileInfo(scriptPath).absolutePath());
        
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("PYTHONIOENCODING", "utf-8");
        trainProcess->setProcessEnvironment(env);
        
        QStringList args;
        args << scriptPath << "--yes";
        if (chkDet->isChecked()) args << "--det";
        if (chkSeg->isChecked()) args << "--seg";
        if (chkTrack->isChecked()) args << "--track";

        // Allow pinning a specific interpreter/venv; fall back to "python" on PATH.
        QString pythonExe = qEnvironmentVariable("RECONSTRUCTION_PYTHON");
        if (pythonExe.isEmpty()) pythonExe = QStringLiteral("python");

        trainProcess->start(pythonExe, args);
    }
}

void AITrainDockWidget::onStopTrain() {
    if (trainProcess->state() != QProcess::NotRunning) {
        trainProcess->kill();
        trainLogBrowser->append("<font color='#d32f2f'>" + m_ctx->translate("aiproc.train_stopping") + "</font>");
        btnStopTrain->setEnabled(false);
        trainProgressBar->hide();
    }
}

void AITrainDockWidget::closeTrainDock() {
    if (trainProcess->state() != QProcess::NotRunning) {
        if (!ModernMessageBox::question(m_ctx->services()->getRequired<ILanguageService>(), m_ctx->shell()->dialogParent(), m_ctx->translate("common.warning"), m_ctx->translate("aiproc.train_running_warn"))) {
            return;
        }
        trainProcess->kill();
    }
    hide();
}

void AITrainDockWidget::onTrainProcessOutput() {
    const QByteArray out = trainProcess->readAllStandardOutput();
    const QByteArray err = trainProcess->readAllStandardError();

    if (!out.isEmpty()) {
        static const QRegularExpression pctRe(QStringLiteral("pct=(\\d{1,3})"));
        const QStringList lines = QString::fromUtf8(out).split('\n', Qt::SkipEmptyParts);
        for (const QString& line : lines) {
            const int marker = line.indexOf(QStringLiteral("[PROGRESS]"));
            if (marker >= 0) {
                const QRegularExpressionMatch match = pctRe.match(line.mid(marker));
                if (match.hasMatch()) {
                    const int pct = qBound(0, match.captured(1).toInt(), 100);
                    if (trainProgressBar->maximum() != 100) trainProgressBar->setRange(0, 100);
                    trainProgressBar->setValue(pct);
                }
            }
            trainLogBrowser->append(line);
        }
    }
    if (!err.isEmpty()) trainLogBrowser->append("<font color='#f59e0b'>" + QString::fromUtf8(err).trimmed() + "</font>");
    
    QScrollBar *sb = trainLogBrowser->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void AITrainDockWidget::onTrainProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    trainProgressBar->hide();
    btnStopTrain->hide();
    if (exitStatus == QProcess::CrashExit) {
        trainLogBrowser->append("<font color='#d32f2f'>" + m_ctx->translate("aiproc.train_crashed") + "</font>");
    } else if (exitCode != 0) {
        trainLogBrowser->append("<font color='#d32f2f'>" + m_ctx->translate("aiproc.train_error").arg(exitCode) + "</font>");
    } else {
        trainLogBrowser->append("<font color='#10a37f'>" + m_ctx->translate("aiproc.train_success") + "</font>");
        ModernMessageBox::information(m_ctx->shell()->dialogParent(), m_ctx->translate("common.success"), m_ctx->translate("aiproc.train_success"));
    }
}
