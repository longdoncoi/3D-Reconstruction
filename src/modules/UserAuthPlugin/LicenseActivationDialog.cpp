#include "LicenseActivationDialog.h"
#include "IUserManager.h"
#include "UserAuthPlugin.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QRadioButton>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QFrame>
#include <QTimer>
#include <QApplication>

LicenseActivationDialog::LicenseActivationDialog(const QString &username,
                                                  IUserManager *users,
                                                  bool allowClose, QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("license.title"), parent), m_username(username),
      m_allowClose(allowClose), m_users(users)
{
    setMinimumWidth(480);
    setupUi();
    applyStyle();
}

void LicenseActivationDialog::setupUi() {
    auto *content = new QWidget(this);
    auto *vl = new QVBoxLayout(content);
    vl->setContentsMargins(20, 20, 20, 20);
    vl->setSpacing(16);

    // Header
    auto *icon = new QLabel("🚀", content);
    icon->setAlignment(Qt::AlignCenter);
    icon->setStyleSheet("font-size: 36px;");
    vl->addWidget(icon);

    auto *title = new QLabel(UserAuthPlugin::translate("license.welcome"), content);
    title->setObjectName("dlgTitle");
    title->setAlignment(Qt::AlignCenter);
    vl->addWidget(title);

    auto *desc = new QLabel(UserAuthPlugin::translate("license.desc"), content);
    desc->setObjectName("descLabel");
    desc->setWordWrap(true);
    desc->setAlignment(Qt::AlignCenter);
    vl->addWidget(desc);

    // Radio options
    auto *btnGroup = new QButtonGroup(content);
    auto *optFrame = new QFrame(content);
    optFrame->setObjectName("optFrame");
    auto *optLayout = new QVBoxLayout(optFrame);
    optLayout->setSpacing(12);

    m_trialRadio = new QRadioButton("🕐  " + UserAuthPlugin::translate("license.trial_btn"), optFrame);
    m_trialRadio->setObjectName("radioOpt");
    m_trialRadio->setChecked(true);
    btnGroup->addButton(m_trialRadio);
    optLayout->addWidget(m_trialRadio);

    m_keyRadio = new QRadioButton("🔑  " + UserAuthPlugin::translate("license.key_label"), optFrame);
    m_keyRadio->setObjectName("radioOpt");
    btnGroup->addButton(m_keyRadio);
    optLayout->addWidget(m_keyRadio);

    m_keyEdit = new QLineEdit(optFrame);
    m_keyEdit->setObjectName("inputField");
    m_keyEdit->setInputMask(">NNNNN-NNNNN-NNNNN-NNNNN;#");
    m_keyEdit->setFixedHeight(40);
    m_keyEdit->setVisible(false);
    m_keyEdit->setAlignment(Qt::AlignCenter);
    m_keyEdit->setPlaceholderText("XXXXX-XXXXX-XXXXX-XXXXX");
    optLayout->addWidget(m_keyEdit);

    vl->addWidget(optFrame);

    m_statusLabel = new QLabel(content);
    m_statusLabel->setObjectName("statusLabel");
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setVisible(false);
    vl->addWidget(m_statusLabel);

    m_activateBtn = new QPushButton("✅  " + UserAuthPlugin::translate("license.activate_btn"), content);
    m_activateBtn->setObjectName("primary");
    m_activateBtn->setFixedHeight(42);
    m_activateBtn->setCursor(Qt::PointingHandCursor);
    m_activateBtn->setDefault(true);
    vl->addWidget(m_activateBtn);

    connect(m_trialRadio,  &QRadioButton::toggled, this, &LicenseActivationDialog::onRadioChanged);
    connect(m_keyRadio,    &QRadioButton::toggled, this, &LicenseActivationDialog::onRadioChanged);
    connect(m_activateBtn, &QPushButton::clicked, this, &LicenseActivationDialog::onActivate);

    setContentLayout(vl);
}

void LicenseActivationDialog::applyStyle() {
    setStyleSheet(R"(
        #dlgTitle { font-size: 20px; font-weight: bold; color: #f1f5f9; }
        #descLabel { color: #94a3b8; font-size: 13px; }
        #optFrame {
            background: #161e2e; border: 1px solid #1e293b;
            border-radius: 12px; padding: 15px;
        }
        #radioOpt { color: #f1f5f9; font-size: 14px; spacing: 10px; }
        #statusLabel { font-size: 13px; padding: 5px; border-radius: 4px; }
    )");
}

void LicenseActivationDialog::onRadioChanged() {
    m_keyEdit->setVisible(m_keyRadio->isChecked());
    m_statusLabel->setVisible(false);
}

void LicenseActivationDialog::setStatus(const QString &msg, bool ok) {
    m_statusLabel->setText(msg);
    m_statusLabel->setStyleSheet(ok ? "color: #4ade80; background: rgba(74,222,128,0.1);" 
                                    : "color: #f87171; background: rgba(248,113,113,0.1);");
    m_statusLabel->setVisible(true);
}

void LicenseActivationDialog::onActivate() {
    IUserManager *um = m_users;
    bool success = false;
    QString msg;
    AuthError code = AuthError::Unknown;

    if (m_trialRadio->isChecked()) {
        success = um->activateTrial(m_username, msg, &code);
    } else {
        QString key = m_keyEdit->text();
        if (key.length() < 20) {
            setStatus(UserAuthPlugin::translate("license.err_length"), false);
            return;
        }
        success = um->activateLicenseKey(m_username, key, msg, &code);
    }

    // Branch on the structured code — never on the Vietnamese message text.
    QString transMsg;
    if (success) {
        transMsg = UserAuthPlugin::translate("license.success_msg");
    } else {
        const QString key = IUserManager::authErrorKey(code);
        transMsg = key.isEmpty() ? msg : UserAuthPlugin::translate(key);
    }

    setStatus(transMsg, success);
    if (success) {
        m_activateBtn->setEnabled(false);
        QTimer::singleShot(1500, this, &QDialog::accept);
    }
}
