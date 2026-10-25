#include "LoginDialog.h"
#include "IUserManager.h"
#include "UserAuthPlugin.h"
#include "ForgotPasswordDialog.h"
#include "ChangePasswordDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QGraphicsDropShadowEffect>

LoginDialog::LoginDialog(IUserManager *users, QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("login.title"), parent)
    , m_users(users)
{
    setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);
    setFixedSize(400, 480);
    setupUi();
    applyStyle();

    // Restore remember me state
    auto *um = m_users;
    if (um->isRememberMeEnabled()) {
        m_userEdit->setText(um->savedUsername());
        m_rememberCheck->setChecked(true);
    }
}

void LoginDialog::setupUi() {
    auto *content = new QWidget(this);
    auto *vl = new QVBoxLayout(content);
    vl->setContentsMargins(30, 30, 30, 30);
    vl->setSpacing(15);

    auto *logo = new QLabel("👤", content);
    logo->setAlignment(Qt::AlignCenter);
    logo->setStyleSheet("font-size: 48px; margin-bottom: 10px;");
    vl->addWidget(logo);

    auto addLabel = [&](const QString &txt) {
        auto *l = new QLabel(txt, content);
        l->setStyleSheet("color: #94a3b8; font-size: 13px;");
        vl->addWidget(l);
    };

    addLabel(UserAuthPlugin::translate("login.username_label"));
    m_userEdit = new QLineEdit(content);
    m_userEdit->setPlaceholderText(
        UserAuthPlugin::translate("common.username_or_email"));
    m_userEdit->setFixedHeight(42);
    vl->addWidget(m_userEdit);

    addLabel(UserAuthPlugin::translate("login.password_label"));
    m_passEdit = new QLineEdit(content);
    m_passEdit->setPlaceholderText(
        UserAuthPlugin::translate("common.password"));
    m_passEdit->setEchoMode(QLineEdit::Password);
    m_passEdit->setFixedHeight(42);
    vl->addWidget(m_passEdit);

    auto *optionsLayout = new QHBoxLayout();
    m_rememberCheck = new QCheckBox(UserAuthPlugin::translate("login.remember_me"), content);
    m_rememberCheck->setCursor(Qt::PointingHandCursor);
    optionsLayout->addWidget(m_rememberCheck);
    optionsLayout->addStretch();

    auto *linksLayout = new QVBoxLayout();
    linksLayout->setSpacing(4);
    linksLayout->setAlignment(Qt::AlignRight);

    auto *forgotBtn = new QPushButton(UserAuthPlugin::translate("login.forgot_pass"), content);
    forgotBtn->setFlat(true);
    forgotBtn->setCursor(Qt::PointingHandCursor);
    forgotBtn->setStyleSheet("QPushButton { color: #818cf8; border: none; background: transparent; padding: 0; text-align: right; } "
                             "QPushButton:hover { color: #a5b4fc; text-decoration: underline; }");
    linksLayout->addWidget(forgotBtn);

    auto *changePwBtn = new QPushButton(UserAuthPlugin::translate("login.change_pass"), content);
    changePwBtn->setFlat(true);
    changePwBtn->setCursor(Qt::PointingHandCursor);
    changePwBtn->setStyleSheet("QPushButton { color: #818cf8; border: none; background: transparent; padding: 0; text-align: right; } "
                               "QPushButton:hover { color: #a5b4fc; text-decoration: underline; }");
    linksLayout->addWidget(changePwBtn);

    optionsLayout->addLayout(linksLayout);
    vl->addLayout(optionsLayout);

    m_statusLabel = new QLabel(content);
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setVisible(false);
    vl->addWidget(m_statusLabel);

    vl->addStretch();

    auto *loginBtn = new QPushButton(UserAuthPlugin::translate("login.btn"), content);
    loginBtn->setObjectName("primary");
    loginBtn->setFixedHeight(46);
    loginBtn->setCursor(Qt::PointingHandCursor);
    loginBtn->setDefault(true);
    vl->addWidget(loginBtn);

    connect(loginBtn, &QPushButton::clicked, this, &LoginDialog::onLogin);
    connect(forgotBtn, &QPushButton::clicked, this, &LoginDialog::onForgotPassword);
    connect(changePwBtn, &QPushButton::clicked, this, &LoginDialog::onChangePassword);

    setContentLayout(vl);
}

void LoginDialog::applyStyle() {
    setStyleSheet("#statusLabel { font-size: 13px; padding: 8px; border-radius: 6px; }");
}

void LoginDialog::setError(const QString &msg) {
    m_statusLabel->setText(msg);
    m_statusLabel->setStyleSheet("color: #f87171; background: rgba(248,113,113,0.1);");
    m_statusLabel->setVisible(true);
}

void LoginDialog::onLogin() {
    QString user = m_userEdit->text().trimmed();
    QString pass = m_passEdit->text();

    if (user.isEmpty() || pass.isEmpty()) {
        setError(UserAuthPlugin::translate("login.err_empty"));
        return;
    }

    QString error;
    AuthError code = AuthError::Unknown;
    if (m_users->login(user, pass, error, &code)) {
        m_loggedInUser = m_users->currentUser().username;
        
        // Handle remember me
        m_users->setRememberMe(m_rememberCheck->isChecked(), user);
        
        accept();
    } else {
        // Branch on the structured code — never on the Vietnamese message text.
        const QString key = IUserManager::authErrorKey(code);
        setError(key.isEmpty()
                     ? (error.isEmpty() ? UserAuthPlugin::translate("login.err_wrong") : error)
                     : UserAuthPlugin::translate(key));
    }
}

void LoginDialog::onForgotPassword() {
    ForgotPasswordDialog dlg(m_users, this);
    dlg.exec();
}

void LoginDialog::onChangePassword() {
    QString user = m_userEdit->text().trimmed();
    ChangePasswordDialog dlg(user, m_users, this);
    dlg.exec();
}
