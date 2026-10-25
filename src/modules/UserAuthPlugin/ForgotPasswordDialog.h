#ifndef FORGOT_PASSWORD_DIALOG_H
#define FORGOT_PASSWORD_DIALOG_H

#include "../../utils/BlockingCall.h"
#include "../../utils/ModernDialog.h"

class QLineEdit;
class QLabel;
class QPushButton;
class QStackedWidget;
class IUserManager;

class ForgotPasswordDialog : public ModernDialog {
    Q_OBJECT
public:
    /// `users` is injected by the owner (registry-resolved). See AboutDialog.
    explicit ForgotPasswordDialog(IUserManager *users, QWidget *parent = nullptr);
    ~ForgotPasswordDialog() override;

private slots:
    void onSendOtp();
    void onVerifyOtp();
    void onResetPassword();

private:
    void setupUi();
    void applyStyle();
    void setError(const QString &msg);

    QStackedWidget *m_stack;

    /// The background tasks of this dialog (the OTP request). The destructor
    /// waits for them: a task must not outlive the service it calls.
    blocking::TaskTracker m_tasks;

    // Step 1 – Enter email
    QLineEdit   *m_emailEdit;
    QPushButton *m_sendOtpBtn;

    // Step 2 – Enter OTP
    QLineEdit   *m_otpEdit;
    QPushButton *m_verifyOtpBtn;

    // Step 3 – New password
    QLineEdit   *m_newPassEdit;
    QLineEdit   *m_confirmPassEdit;
    QPushButton *m_resetBtn;

    QLabel  *m_statusLabel;
    QString  m_pendingEmail;
    IUserManager *m_users = nullptr;
};

#endif // FORGOT_PASSWORD_DIALOG_H
