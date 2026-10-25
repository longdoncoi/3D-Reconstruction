#include "AboutDialog.h"
#include "IUserManager.h"
#include "UserAuthPlugin.h"
#include "LicenseActivationDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFrame>

AboutDialog::AboutDialog(const QString &username, IUserManager *users, QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("about.title"), parent), m_username(username),
      m_users(users)
{
    setMinimumWidth(440);
    setupUi();
    applyStyle();
}

void AboutDialog::setupUi() {
    auto *um = m_users;
    UserInfo u = um->userInfo(m_username);
    int daysLeft = um->daysRemaining(m_username);
    QString status = um->licenseStatusStr(m_username);

    auto *content = new QWidget(this);
    auto *vl = new QVBoxLayout(content);
    vl->setContentsMargins(20, 20, 20, 20);
    vl->setSpacing(14);

    // App logo/title
    auto *appIcon = new QLabel("🔬", content);
    appIcon->setAlignment(Qt::AlignCenter);
    appIcon->setStyleSheet("font-size: 42px;");
    vl->addWidget(appIcon);

    auto *appName = new QLabel("3D Reconstruction Studio", content);
    appName->setObjectName("appName");
    appName->setAlignment(Qt::AlignCenter);
    vl->addWidget(appName);

    auto *version = new QLabel("Version 1.0.0  |  Build 2026", content);
    version->setObjectName("versionLabel");
    version->setAlignment(Qt::AlignCenter);
    vl->addWidget(version);

    // User info card
    auto *userSection = new QFrame(content);
    userSection->setObjectName("infoCard");
    auto *gl = new QVBoxLayout(userSection);
    gl->setSpacing(8);

    auto addRow = [&](const QString &label, const QString &value, const QString &valueColor = "#fff") {
        auto *hl = new QHBoxLayout;
        auto *lbl = new QLabel(label + ":", userSection);
        lbl->setObjectName("rowLabel");
        lbl->setFixedWidth(120);
        auto *val = new QLabel(value, userSection);
        val->setObjectName("rowValue");
        val->setStyleSheet("color: " + valueColor + ";");
        hl->addWidget(lbl);
        hl->addWidget(val, 1);
        gl->addLayout(hl);
    };

    addRow(UserAuthPlugin::translate("admin.col_username"), u.username);
    addRow(UserAuthPlugin::translate("admin.col_email"), u.email);
    addRow(UserAuthPlugin::translate("admin.col_role"), u.role == UserRole::Admin ? "🛡 Admin" : UserAuthPlugin::translate("admin.role_user"));

    QString statusIcon, statusColor;
    if (status == "Activated") {
        statusIcon = UserAuthPlugin::translate("about.activated");
        statusColor = "#4ade80";
    } else if (status == "Trial") {
        statusIcon = UserAuthPlugin::translate("about.trial");
        statusColor = "#fbbf24";
    } else {
        statusIcon = UserAuthPlugin::translate("about.not_activated");
        statusColor = "#f87171";
    }
    addRow(UserAuthPlugin::translate("admin.col_status"), statusIcon, statusColor);

    if (status != "None") {
        QString expiryStr = u.licenseExpiry.isValid() ? u.licenseExpiry.toString("dd/MM/yyyy") : "—";
        addRow(UserAuthPlugin::translate("admin.col_expiry"), expiryStr);
        addRow(UserAuthPlugin::translate("about.remaining"),  QString::number(daysLeft) + UserAuthPlugin::translate("about.days_suffix"), daysLeft > 10 ? "#4ade80" : "#f87171");
    }

    vl->addWidget(userSection);

    if (status == "Trial") {
        auto *activateLink = new QPushButton("🔑 " + UserAuthPlugin::translate("about.upgrade_btn"), content);
        activateLink->setObjectName("activateLink");
        activateLink->setCursor(Qt::PointingHandCursor);
        activateLink->setFlat(true);
        vl->addWidget(activateLink, 0, Qt::AlignCenter);
        connect(activateLink, &QPushButton::clicked, this, &AboutDialog::onActivateLicense);
    }

    vl->addStretch();

    auto *closeBtn = new QPushButton(UserAuthPlugin::translate("common.close"), content);
    closeBtn->setObjectName("primary"); // Use StyleManager primary style
    closeBtn->setFixedHeight(38);
    vl->addWidget(closeBtn);
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    setContentLayout(vl);
}

void AboutDialog::applyStyle() {
    // Keep app-specific text styles
    setStyleSheet(R"(
        #appName { font-size: 20px; font-weight: bold; color: #a78bfa; }
        #versionLabel { color: #666; font-size: 12px; }
        #infoCard {
            background: #161e2e; border: 1px solid #1e293b;
            border-radius: 10px; padding: 10px;
        }
        #rowLabel { color: #94a3b8; font-size: 13px; }
        #rowValue { color: #f1f5f9; font-size: 13px; font-weight: 500; }
        #activateLink {
            color: #818cf8; font-size: 13px; font-weight: bold;
            text-decoration: underline; border: none; background: transparent;
        }
        #activateLink:hover { color: #a5b4fc; }
    )");
}

void AboutDialog::onActivateLicense() {
    LicenseActivationDialog dlg(m_username, m_users, true, this);
    if (dlg.exec() == QDialog::Accepted) {
        accept();
    }
}
