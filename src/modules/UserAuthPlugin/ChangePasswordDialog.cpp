#include "ChangePasswordDialog.h"
#include "IUserManager.h"
#include "UserAuthPlugin.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QTimer>

ChangePasswordDialog::ChangePasswordDialog(const QString &username, IUserManager *users,
                                           QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("changepw.title"), parent), m_username(username),
      m_users(users)
{
    m_isExternal = !m_users->isLoggedIn();
    if (m_isExternal) {
        setFixedSize(380, 500);
    } else {
        setFixedSize(380, 420);
    }
    setupUi();
    applyStyle();
}

void ChangePasswordDialog::setupUi() {
    auto *content = new QWidget(this);
    auto *vl = new QVBoxLayout(content);
    vl->setContentsMargins(20, 20, 20, 20);
    vl->setSpacing(12);

    auto addLabel = [&](const QString &txt) {
        auto *l = new QLabel(txt, content);
        l->setStyleSheet("color: #94a3b8; font-size: 13px;");
        vl->addWidget(l);
        return l;
    };

    if (m_isExternal) {
        addLabel(UserAuthPlugin::translate("login.username_label"));
        m_userEdit = new QLineEdit(content);
        m_userEdit->setPlaceholderText(
            UserAuthPlugin::translate("common.username_or_email"));
        m_userEdit->setText(m_username);
        m_userEdit->setFixedHeight(40);
        vl->addWidget(m_userEdit);
    }

    addLabel(UserAuthPlugin::translate("changepw.old"));
    m_oldPassEdit = new QLineEdit(content);
    m_oldPassEdit->setEchoMode(QLineEdit::Password);
    m_oldPassEdit->setFixedHeight(40);
    vl->addWidget(m_oldPassEdit);

    addLabel(UserAuthPlugin::translate("changepw.new"));
    m_newPassEdit = new QLineEdit(content);
    m_newPassEdit->setEchoMode(QLineEdit::Password);
    m_newPassEdit->setFixedHeight(40);
    vl->addWidget(m_newPassEdit);

    addLabel(UserAuthPlugin::translate("changepw.confirm"));
    m_confirmEdit = new QLineEdit(content);
    m_confirmEdit->setEchoMode(QLineEdit::Password);
    m_confirmEdit->setFixedHeight(40);
    vl->addWidget(m_confirmEdit);

    m_statusLabel = new QLabel(content);
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setVisible(false);
    vl->addWidget(m_statusLabel);

    vl->addStretch();

    m_applyBtn = new QPushButton(UserAuthPlugin::translate("changepw.btn"), content);
    m_applyBtn->setObjectName("primary");
    m_applyBtn->setFixedHeight(44);
    m_applyBtn->setCursor(Qt::PointingHandCursor);
    m_applyBtn->setDefault(true);
    vl->addWidget(m_applyBtn);

    connect(m_applyBtn, &QPushButton::clicked, this, &ChangePasswordDialog::onApply);

    setContentLayout(vl);
}

void ChangePasswordDialog::applyStyle() {
    setStyleSheet("#statusLabel { font-size: 13px; padding: 5px; border-radius: 4px; }");
}

void ChangePasswordDialog::setStatus(const QString &msg, bool ok) {
    m_statusLabel->setText(msg);
    m_statusLabel->setStyleSheet(ok ? "color: #4ade80; background: rgba(74,222,128,0.1);" 
                                    : "color: #f87171; background: rgba(248,113,113,0.1);");
    m_statusLabel->setVisible(true);
}

void ChangePasswordDialog::onApply() {
    QString targetUser = m_username;
    if (m_isExternal) {
        if (!m_userEdit) return;
        targetUser = m_userEdit->text().trimmed();
        if (targetUser.isEmpty()) {
            setStatus(UserAuthPlugin::translate("forgot.err_empty_email"), false);
            return;
        }
        if (targetUser.contains('@')) {
            QString resolved = m_users->findUsernameByEmail(targetUser);
            if (!resolved.isEmpty()) {
                targetUser = resolved;
            }
        }
    }

    QString oldP = m_oldPassEdit->text();
    QString newP = m_newPassEdit->text();
    QString conf = m_confirmEdit->text();

    if (oldP.isEmpty() || newP.isEmpty()) {
        setStatus(UserAuthPlugin::translate("login.err_empty"), false);
        return;
    }
    if (newP != conf) {
        setStatus(UserAuthPlugin::translate("changepw.err_mismatch"), false);
        return;
    }

    QString msg;
    AuthError code = AuthError::Unknown;
    if (m_users->changePassword(targetUser, oldP, newP, msg, &code)) {
        setStatus(UserAuthPlugin::translate("changepw.success"), true);
        m_applyBtn->setEnabled(false);
        QTimer::singleShot(1500, this, &QDialog::accept);
    } else {
        // Branch on the structured code — never on the Vietnamese message text.
        const QString key = IUserManager::authErrorKey(code);
        setStatus(key.isEmpty() ? msg : UserAuthPlugin::translate(key), false);
    }
}
