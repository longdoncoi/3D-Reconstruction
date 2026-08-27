#include "AITrainDockWidget.h"
#include "IAppContext.h"
#include "AppConfig.h"
#include "../../utils/ModernMessageBox.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFile>
#include <QFileInfo>
#include <QScrollBar>
#include <QDialog>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QTimer>

AITrainDockWidget::AITrainDockWidget(IAppContext* ctx, QWidget* parent)
    : QDockWidget("🚀 AI Training", parent), m_ctx(ctx) 
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
    
    QLabel *titleLabel = new QLabel("🚀 AI Training", topBar);
    titleLabel->setStyleSheet("color:#ddd; font-weight:bold; font-size:14px;");

    trainProgressBar = new QProgressBar(topBar);
    trainProgressBar->setTextVisible(false);
    trainProgressBar->setStyleSheet(
        "QProgressBar { background:#2a2a35; border:1px solid #3a3a4a; border-radius:6px; color:#fff; text-align:center; height: 24px; }"
        "QProgressBar::chunk { background-color: #10a37f; border-radius:5px; }"
    );
    trainProgressBar->setRange(0, 0);
    trainProgressBar->hide();
    
    btnStopTrain = new QPushButton("Stop", topBar);
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
    
    tl->addWidget(titleLabel, 0);
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

    m_ctx->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, this);
    hide();

    trainProcess = new QProcess(this);
    
    connect(btnCloseTrain, &QPushButton::clicked, this, &AITrainDockWidget::closeTrainDock);
    connect(btnStopTrain, &QPushButton::clicked, this, &AITrainDockWidget::onStopTrain);
    connect(trainProcess, &QProcess::readyReadStandardOutput, this, &AITrainDockWidget::onTrainProcessOutput);
    connect(trainProcess, &QProcess::readyReadStandardError, this, &AITrainDockWidget::onTrainProcessOutput);
    connect(trainProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &AITrainDockWidget::onTrainProcessFinished);
}

AITrainDockWidget::~AITrainDockWidget() {
    if (trainProcess->state() != QProcess::NotRunning) {
        trainProcess->kill();
        trainProcess->waitForFinished();
    }
}

void AITrainDockWidget::startTraining(bool autoConfirm) {
    show();
    if (trainProcess->state() == QProcess::NotRunning) {
        QString scriptPath = AppConfig::instance().aiComputerVisionDir() + "/TrainModel.py";
        QString modelsPath = AppConfig::instance().modelsDir();
        bool modelExists = QFile::exists(modelsPath + "/yolo11n.onnx") || QFile::exists(modelsPath + "/yolo11n-seg.onnx") || QFile::exists(modelsPath + "/yolo11n-tracking.onnx");
        
        QDialog dialog(m_ctx->mainWindow());
        dialog.setWindowTitle(m_ctx->translate("ai.training"));
        dialog.setStyleSheet("QDialog { background-color: #1a1a1f; color: white; } QCheckBox { color: white; font-size: 14px; margin: 5px; }");
        
        QVBoxLayout *layout = new QVBoxLayout(&dialog);
        QLabel* lbl = new QLabel("Select models to train:", &dialog);
        lbl->setStyleSheet("color: white; font-size: 14px; font-weight: bold;");
        layout->addWidget(lbl);
        
        QCheckBox *chkDet = new QCheckBox("Detection (yolo11n)", &dialog);
        QCheckBox *chkSeg = new QCheckBox("Segmentation (yolo11n-seg)", &dialog);
        QCheckBox *chkTrack = new QCheckBox("Tracking (yolo11x-tracking)", &dialog);
        
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

        if (modelExists && !autoConfirm) {
            if (!ModernMessageBox::question(m_ctx->mainWindow(), m_ctx->translate("aiproc.confirm"), m_ctx->translate("aiproc.model_exists_retrain"))) {
                return;
            }
        }
    
        trainLogBrowser->clear();
        trainLogBrowser->append("<font color='#10a37f'>--- Bắt đầu tiến trình huấn luyện ---</font>");
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

        trainProcess->start("python", args);
    }
}

void AITrainDockWidget::onStopTrain() {
    if (trainProcess->state() != QProcess::NotRunning) {
        trainProcess->kill();
        trainLogBrowser->append("<font color='#d32f2f'>Đang dừng tiến trình...</font>");
        btnStopTrain->setEnabled(false);
        trainProgressBar->hide();
    }
}

void AITrainDockWidget::closeTrainDock() {
    if (trainProcess->state() != QProcess::NotRunning) {
        if (!ModernMessageBox::question(m_ctx->mainWindow(), m_ctx->translate("aiproc.warning"), m_ctx->translate("aiproc.train_running_warn"))) {
            return;
        }
        trainProcess->kill();
    }
    hide();
}

void AITrainDockWidget::onTrainProcessOutput() {
    QByteArray out = trainProcess->readAllStandardOutput();
    QByteArray err = trainProcess->readAllStandardError();
    if (!out.isEmpty()) trainLogBrowser->append(QString::fromUtf8(out).trimmed());
    if (!err.isEmpty()) trainLogBrowser->append("<font color='#f59e0b'>" + QString::fromUtf8(err).trimmed() + "</font>");
    
    QScrollBar *sb = trainLogBrowser->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void AITrainDockWidget::onTrainProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    trainProgressBar->hide();
    btnStopTrain->hide();
    if (exitStatus == QProcess::CrashExit) {
        trainLogBrowser->append("<font color='#d32f2f'>Tiến trình bị gián đoạn (Crash).</font>");
    } else if (exitCode != 0) {
        trainLogBrowser->append(QString("<font color='#d32f2f'>Có lỗi xảy ra (Mã lỗi: %1).</font>").arg(exitCode));
    } else {
        trainLogBrowser->append("<font color='#10a37f'>Huấn luyện thành công!</font>");
        ModernMessageBox::information(m_ctx->mainWindow(), m_ctx->translate("aiproc.success"), m_ctx->translate("aiproc.train_success"));
    }
}
