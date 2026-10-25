#include "BlockingCall.h"
#include "IMailService.h"
#include "ForgotPasswordDialog.h"
#include "IUserManager.h"
#include "UserAuthPlugin.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QTimer>
#include "../../utils/ModernMessageBox.h"
#include <QFutureWatcher>
#include <QDebug>
#include <QRegularExpression>

ForgotPasswordDialog::ForgotPasswordDialog(IUserManager *users, QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("forgot.title"), parent)
    , m_users(users)
{
    setFixedSize(400, 350);
    setupUi();
    applyStyle();
}

ForgotPasswordDialog::~ForgotPasswordDialog()
{
    // The OTP request may still be running: it must not outlive the dialog
    // (and the service it calls). The wait is bounded.
    const int unfinished = m_tasks.waitForFinished(3000);
    if (unfinished > 0) {
        qWarning() << "ForgotPasswordDialog:" << unfinished
                   << "background task(s) did not finish within 3 s";
    }
}

void ForgotPasswordDialog::setupUi() {
    auto *content = new QWidget(this);
    auto *vl = new QVBoxLayout(content);
    vl->setContentsMargins(20, 20, 20, 20);
    vl->setSpacing(15);

    m_stack = new QStackedWidget(content);
    
    // Step 1: Email
    auto *step1 = new QWidget;
    auto *s1l = new QVBoxLayout(step1);
    s1l->setContentsMargins(0, 0, 0, 0);
    s1l->addWidget(new QLabel(UserAuthPlugin::translate("forgot.email_label"), step1));
    m_emailEdit = new QLineEdit(step1);
    m_emailEdit->setPlaceholderText("example@mail.com");
    m_emailEdit->setFixedHeight(40);
    s1l->addWidget(m_emailEdit);
    m_sendOtpBtn = new QPushButton(UserAuthPlugin::translate("forgot.send_otp"), step1);
    m_sendOtpBtn->setObjectName("primary");
    m_sendOtpBtn->setFixedHeight(40);
    s1l->addWidget(m_sendOtpBtn);
    s1l->addStretch();
    m_stack->addWidget(step1);

    // Step 2: OTP
    auto *step2 = new QWidget;
    auto *s2l = new QVBoxLayout(step2);
    s2l->setContentsMargins(0, 0, 0, 0);
    s2l->addWidget(new QLabel(UserAuthPlugin::translate("forgot.otp_label"), step2));
    m_otpEdit = new QLineEdit(step2);
    m_otpEdit->setPlaceholderText("OTP");
    m_otpEdit->setFixedHeight(40);
    s2l->addWidget(m_otpEdit);
    m_verifyOtpBtn = new QPushButton(UserAuthPlugin::translate("common.ok"), step2);
    m_verifyOtpBtn->setObjectName("primary");
    m_verifyOtpBtn->setFixedHeight(40);
    m_verifyOtpBtn->setDefault(true);
    s2l->addWidget(m_verifyOtpBtn);
    s2l->addStretch();
    m_stack->addWidget(step2);

    // Step 3: New Password
    auto *step3 = new QWidget;
    auto *s3l = new QVBoxLayout(step3);
    s3l->setContentsMargins(0, 0, 0, 0);
    s3l->addWidget(new QLabel(UserAuthPlugin::translate("forgot.new_pass"), step3));
    m_newPassEdit = new QLineEdit(step3);
    m_newPassEdit->setEchoMode(QLineEdit::Password);
    m_newPassEdit->setFixedHeight(40);
    s3l->addWidget(m_newPassEdit);
    s3l->addWidget(new QLabel(UserAuthPlugin::translate("changepw.confirm"), step3));
    m_confirmPassEdit = new QLineEdit(step3);
    m_confirmPassEdit->setEchoMode(QLineEdit::Password);
    m_confirmPassEdit->setFixedHeight(40);
    s3l->addWidget(m_confirmPassEdit);
    m_resetBtn = new QPushButton(UserAuthPlugin::translate("forgot.reset_btn"), step3);
    m_resetBtn->setObjectName("primary");
    m_resetBtn->setFixedHeight(40);
    s3l->addWidget(m_resetBtn);
    s3l->addStretch();
    m_stack->addWidget(step3);

    vl->addWidget(m_stack);

    m_statusLabel = new QLabel(content);
    m_statusLabel->setObjectName("statusLabel");
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setVisible(false);
    vl->addWidget(m_statusLabel);

    connect(m_sendOtpBtn,   &QPushButton::clicked, this, &ForgotPasswordDialog::onSendOtp);
    connect(m_verifyOtpBtn, &QPushButton::clicked, this, &ForgotPasswordDialog::onVerifyOtp);
    connect(m_resetBtn,     &QPushButton::clicked, this, &ForgotPasswordDialog::onResetPassword);

    setContentLayout(vl);
}

void ForgotPasswordDialog::applyStyle() {
    setStyleSheet("QLabel { color: #94a3b8; font-size: 13px; } "
                  "#statusLabel { font-size: 13px; padding: 5px; border-radius: 4px; }");
}

void ForgotPasswordDialog::setError(const QString &msg) {
    m_statusLabel->setText(msg);
    m_statusLabel->setStyleSheet("color: #f87171; background: rgba(248,113,113,0.1);");
    m_statusLabel->setVisible(true);
}

void ForgotPasswordDialog::onSendOtp() {
    QString email = m_emailEdit->text().trimmed();
    if (email.isEmpty()) {
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), UserAuthPlugin::translate("forgot.err_empty_email"));
        return;
    }
    
    QRegularExpression mailRegex("^[\\w\\.-]+@([\\w-]+\\.)+[\\w-]{2,4}$");
    if (!mailRegex.match(email).hasMatch()) {
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), UserAuthPlugin::translate("forgot.err_invalid_email"));
        return;
    }

    if (m_users->findUsernameByEmail(email).isEmpty()) {
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), UserAuthPlugin::translate("forgot.err_no_email"));
        return;
    }
    
    m_sendOtpBtn->setEnabled(false);
    auto *watcher = new QFutureWatcher<QPair<bool, QString>>(this);
    connect(watcher, &QFutureWatcher<QPair<bool, QString>>::finished, this,
            [this, watcher, email]() {
        const auto result = watcher->result();
        watcher->deleteLater();
        m_sendOtpBtn->setEnabled(true);
        if (result.first) {
            m_pendingEmail = email;
            m_stack->setCurrentIndex(1);
            m_statusLabel->setVisible(false);
        } else {
            setError(result.second);
        }
    });
    // The user manager is captured, not looked up on the worker thread: the
    // plugin's static binding is cleared in cleanup(), so a worker that
    // resolved it late would dereference a null pointer. The captured
    // pointer refers to the process-lifetime service.
    IUserManager *users = m_users;
    watcher->setFuture(blocking::run([users, email]() {
        QString error;
        const bool ok = users->sendPasswordResetOtp(email, error);
        return qMakePair(ok, error);
    }, &m_tasks));
}

void ForgotPasswordDialog::onVerifyOtp() {
    QString otp = m_otpEdit->text().trimmed();
    if (m_users->verifyOtp(m_pendingEmail, otp)) {
        m_stack->setCurrentIndex(2);
        m_statusLabel->setVisible(false);
    } else {
        setError(UserAuthPlugin::translate("forgot.err_otp"));
    }
}

void ForgotPasswordDialog::onResetPassword() {
    QString p1 = m_newPassEdit->text();
    QString p2 = m_confirmPassEdit->text();
    if (p1.isEmpty() || p1 != p2) { setError(UserAuthPlugin::translate("changepw.err_mismatch")); return; }

    // The core resolves the account from the e-mail and consumes the proof of
    // ownership recorded by verifyOtp() — the dialog does not decide who may
    // reset a password.
    QString error;
    AuthError code = AuthError::None;
    if (m_users->resetPassword(m_pendingEmail, p1, error, &code)) {
        m_statusLabel->setText(UserAuthPlugin::translate("changepw.success"));
        m_statusLabel->setStyleSheet("color: #4ade80; background: rgba(74,222,128,0.1);");
        m_statusLabel->setVisible(true);
        m_resetBtn->setEnabled(false);
        QTimer::singleShot(2000, this, &QDialog::accept);
    } else {
        setError(error.isEmpty() ? UserAuthPlugin::translate("forgot.err_reset") : error);
    }
}
