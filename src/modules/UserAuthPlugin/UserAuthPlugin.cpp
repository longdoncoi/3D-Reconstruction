#include "UserAuthPlugin.h"
#include "IAppContext.h"
#include "ISettingsService.h"
#include "UserManager.h"
#include "LanguageManager.h"
#include "StyleManager.h"
#include "SignalBus.h"
#include "IconFactory.h"
#include "ModernMessageBox.h"

#include "LoginDialog.h"
#include "ChangePasswordDialog.h"
#include "LicenseActivationDialog.h"
#include "AdminUserManagerDialog.h"
#include "AboutDialog.h"
#include "ThemeSelectionDialog.h"
#include "SettingsDialog.h"
#include "AvatarCropperDialog.h"
#include "AppConfig.h"

#include <QMenu>
#include <QMenuBar>
#include <QMainWindow>
#include <QApplication>
#include <QDir>
#include <QPainter>
#include <QToolButton>
#include <QFileDialog>
#include <QStandardPaths>
#include <QLabel>
#include <QHBoxLayout>
#include <QStyle>
#include <QGroupBox>

#include "LanguageManager.h"

QString UserAuthPlugin::translate(const QString& key) {
    return LM_TR(key);
}

// ─── Helper: create circular avatar pixmap ────────────────────────────────────
static QPixmap createAvatarPixmap(const QString &username, const QString &avatarPath) {
    if (!avatarPath.isEmpty() && QFile::exists(avatarPath)) {
        QPixmap pm(avatarPath);
        if (!pm.isNull()) return pm.scaled(32, 32, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    }
    QPixmap pixmap(32, 32);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setBrush(QColor("#7c3aed"));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(0, 0, 32, 32);
    painter.setPen(Qt::white);
    painter.setFont(QFont("Segoe UI", 14, QFont::Bold));
    painter.drawText(pixmap.rect(), Qt::AlignCenter, username.isEmpty() ? "?" : username.left(1).toUpper());
    return pixmap;
}

// ─── Initialize ───────────────────────────────────────────────────────────────
void UserAuthPlugin::initialize(IAppContext* context) {
    m_context = context;

    QString configDir = QFileInfo(AppConfig::instance().configPath()).absolutePath();
    QDir().mkpath(configDir);
    QString configPath = configDir + "/Config_User.ini";
    UserManager::instance()->loadConfig(configPath);

    if (!ensureAuthenticated()) {
        QMetaObject::invokeMethod(qApp, "quit", Qt::QueuedConnection);
        return;
    }

    // Load per-user settings (language, theme…) AFTER we know who's logged in
    loadUserSettings();

    checkLicense();
    setupMenus();

    // Connect SignalBus so we retranslate whenever language changes
    connect(m_context->signalBus(), &SignalBus::languageChanged,
            this, &UserAuthPlugin::onLanguageChanged);
}

void UserAuthPlugin::cleanup() {
    UserManager::instance()->saveConfig();
}

// ─── Authentication ───────────────────────────────────────────────────────────
bool UserAuthPlugin::ensureAuthenticated() {
    auto *um = UserManager::instance();
    if (um->isLoggedIn()) {
        m_currentUser = um->currentUsername();
        return true;
    }
    LoginDialog dlg(m_context->mainWindow());
    if (dlg.exec() == QDialog::Accepted) {
        m_currentUser = dlg.loggedInUsername();
        return true;
    }
    return false;
}

void UserAuthPlugin::checkLicense() {
    auto *um = UserManager::instance();
    if (um->needsActivation(m_currentUser)) {
        LicenseActivationDialog dlg(m_currentUser, false, m_context->mainWindow());
        dlg.exec();
    }
    if (um->isLicenseExpired(m_currentUser)) {
        ModernMessageBox::critical(m_context->mainWindow(),
            LM_TR("common.error"),
            LM_TR("auth.expired"));
        QMetaObject::invokeMethod(qApp, "quit", Qt::QueuedConnection);
    }
}

// ─── Per-user settings ────────────────────────────────────────────────────────
void UserAuthPlugin::loadUserSettings() {
    auto *um = UserManager::instance();

    // Language
    QString lang = um->getUserPref(m_currentUser, "language", "vi");
    m_context->setLanguage(lang);

    // Theme (accent color)
    QString theme = um->getUserPref(m_currentUser, "theme", "");
    if (!theme.isEmpty()) {
        StyleManager::setAccentColor(theme);
    }

    // Background image
    bool    useBg     = um->getUserPref(m_currentUser, "use_bg_image", "0") == "1";
    QString bgPath    = um->getUserPref(m_currentUser, "bg_image_path", "");
    int     bgOpacity = um->getUserPref(m_currentUser, "bg_opacity", "30").toInt();

    // Glassmorphism: semi-transparent panels only when bg image is active
    StyleManager::setGlassmorphismEnabled(useBg);
    ThemeSelectionDialog::applyGlobalBackground(m_context->mainWindow(), bgPath, bgOpacity, useBg);

    emit m_context->signalBus()->userChanged(m_currentUser);
}

void UserAuthPlugin::saveUserPref(const QString &key, const QString &value) {
    UserManager::instance()->setUserPref(m_currentUser, key, value);
}

void UserAuthPlugin::setupMenus() {
    QMenuBar* menuBar = m_context->menuBar();
    if (!menuBar) menuBar = m_context->mainWindow()->menuBar();

    UserInfo u = UserManager::instance()->userInfo(m_currentUser);
    bool isAdmin = (u.role == UserRole::Admin);

    // ── Inject Help Buttons into tab.help panel ──────────────────────────────
    if (QWidget* panel = m_context->getTabPanel("tab.help")) {
        auto *layout = qobject_cast<QHBoxLayout*>(panel->layout());
        if (layout) {
            if (!m_groupHelp) {
                m_groupHelp = new QGroupBox(panel);
                m_groupHelp->setObjectName("helpGroup");
                m_groupHelp->setTitle("");

                QVBoxLayout *vbox = new QVBoxLayout(m_groupHelp);
                vbox->setContentsMargins(4, 4, 4, 4);
                vbox->setSpacing(2);

                QHBoxLayout *gLayout = new QHBoxLayout();
                gLayout->setContentsMargins(0, 0, 0, 0);
                gLayout->setSpacing(5);

                m_btnAbout = new QToolButton(m_groupHelp);
                m_btnAbout->setText(m_context->translate("about.title"));
                m_btnAbout->setIcon(IconFactory::createModern("ℹ️", QColor("#3b82f6"), QColor("#2563eb")));
                m_btnAbout->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
                m_btnAbout->setMinimumWidth(80);
                connect(m_btnAbout, &QToolButton::clicked, this, &UserAuthPlugin::onAbout);

                gLayout->addWidget(m_btnAbout);
                vbox->addLayout(gLayout);
                vbox->addStretch();

                QLabel *titleLabel = new QLabel(m_context->translate("menu.help"), m_groupHelp);
                titleLabel->setObjectName("groupTitleLabel");
                titleLabel->setAlignment(Qt::AlignCenter);
                titleLabel->setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;");
                vbox->addWidget(titleLabel);

                layout->insertWidget(layout->count() - 1, m_groupHelp);
            }
        }
    }

    // ── Language Switcher Button ─────────────────────────────────────────────
    if (!m_langBtn) {
        m_langBtn = new QToolButton(menuBar);
        m_langBtn->setObjectName("langSwitchBtn");
        m_langBtn->setPopupMode(QToolButton::InstantPopup);
        m_langBtn->setCursor(Qt::PointingHandCursor);
        m_langBtn->setMinimumWidth(130);
        m_langBtn->setStyleSheet(
            "QToolButton { border-radius: 4px; border: none; background: transparent;"
            " padding: 2px 6px; font-size: 14px; color: #f1f5f9; }"
            "QToolButton:hover { background: rgba(255,255,255,0.1); }"
            "QToolButton::menu-indicator { image: none; }");
    }
    QString flagPath = UserAuthPlugin::translate("flag_icon_path");
    m_langBtn->setIcon(QIcon(flagPath));
    m_langBtn->setIconSize(QSize(24, 16));
    m_langBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_langBtn->setText(" " + LM.displayName() + "  ▾");

    QMenu* langMenu = new QMenu(m_langBtn);
    QAction* viAction = langMenu->addAction(QIcon("f:/PROJECTS/QT/3D-Reconstruction/src/app/flag_vn.png"), "Tiếng Việt");
    QAction* enAction = langMenu->addAction(QIcon("f:/PROJECTS/QT/3D-Reconstruction/src/app/flag_en.png"), "English");
    connect(viAction, &QAction::triggered, this, [this]() {
        m_context->setLanguage("vi");
        saveUserPref("language", "vi");
    });
    connect(enAction, &QAction::triggered, this, [this]() {
        m_context->setLanguage("en");
        saveUserPref("language", "en");
    });
    m_langBtn->setMenu(langMenu);

    // ── Avatar Button ────────────────────────────────────────────────────────
    bool isNewAvatarBtn = false;
    if (!m_avatarBtn) {
        m_avatarBtn = new QToolButton(menuBar);
        m_avatarBtn->setIconSize(QSize(32, 32));
        m_avatarBtn->setStyleSheet(
            "QToolButton { border-radius: 16px; border: 2px solid rgba(255,255,255,0.15);"
            " background: transparent; padding: 0; }"
            "QToolButton:hover { border-color: #6366f1; }"
            "QToolButton::menu-indicator { image: none; }");
        m_avatarBtn->setPopupMode(QToolButton::InstantPopup);
        m_avatarBtn->setCursor(Qt::PointingHandCursor);
        isNewAvatarBtn = true;
    }

    u = UserManager::instance()->userInfo(m_currentUser);
    QString displayName = u.username;
    if (!displayName.isEmpty()) displayName[0] = displayName[0].toUpper();

    m_avatarBtn->setIcon(QIcon(createAvatarPixmap(u.username, u.avatarPath)));

    if (isNewAvatarBtn) {
        QWidget* cornerWidget = new QWidget(menuBar);
        QHBoxLayout* cornerLayout = new QHBoxLayout(cornerWidget);
        cornerLayout->setContentsMargins(0, 0, 8, 0);
        cornerLayout->setSpacing(4);
        cornerLayout->addWidget(m_langBtn);
        cornerLayout->addWidget(m_avatarBtn);
        menuBar->setCornerWidget(cornerWidget, Qt::TopRightCorner);
        cornerWidget->show();
    }

    // ── Avatar dropdown menu ─────────────────────────────────────────────────
    // Order: 👤 Name (disabled) | ── | ⚙️ Cài đặt | ── | 🖼️ Đổi ảnh | 🔑 Đổi mật khẩu | ── | 🚪 Đăng xuất
    QMenu* oldMenu = m_avatarBtn->menu();
    QMenu* avatarMenu = new QMenu(m_avatarBtn);
    avatarMenu->setStyleSheet("QAction { padding: 10px; }");

    // Username header (disabled)
    QAction* userAction = avatarMenu->addAction("👤 " + displayName);
    userAction->setEnabled(false);
    avatarMenu->addSeparator();

    // ⚙️ Cài đặt (opens SettingsDialog)
    QAction *settingsAction = avatarMenu->addAction(
        IconFactory::createModern("⚙️", QColor("#6366f1"), QColor("#4f46e5")),
        m_context->translate("menu.settings"));
    connect(settingsAction, &QAction::triggered, this, &UserAuthPlugin::onOpenSettings);

    avatarMenu->addSeparator();

    // 🖼️ Đổi ảnh đại diện
    QAction *changeAvatarAct = avatarMenu->addAction(
        IconFactory::createModern("🖼️", QColor("#10b981"), QColor("#059669")),
        m_context->translate("avatar.change_avatar"));
    connect(changeAvatarAct, &QAction::triggered, this, &UserAuthPlugin::onChangeAvatar);

    // 🔑 Đổi mật khẩu
    QAction *changePassAct = avatarMenu->addAction(
        IconFactory::createModern("🔑", QColor("#f59e0b"), QColor("#d97706")),
        m_context->translate("avatar.change_pass"));
    connect(changePassAct, &QAction::triggered, this, &UserAuthPlugin::onChangePassword);

    avatarMenu->addSeparator();

    // 🚪 Đăng xuất
    QAction *logoutAct = avatarMenu->addAction(
        IconFactory::createModern("🚪", QColor("#ef4444"), QColor("#dc2626")),
        m_context->translate("avatar.logout"));
    connect(logoutAct, &QAction::triggered, this, &UserAuthPlugin::onLogout);

    m_avatarBtn->setMenu(avatarMenu);
    if (oldMenu) oldMenu->deleteLater();
}

// ─── Retranslate ──────────────────────────────────────────────────────────────
void UserAuthPlugin::retranslateMenus() {
    if (m_langBtn) {
        QString flagPath = LM.currentLanguage() == "vi"
            ? "f:/PROJECTS/QT/3D-Reconstruction/src/app/flag_vn.png"
            : "f:/PROJECTS/QT/3D-Reconstruction/src/app/flag_en.png";
        m_langBtn->setIcon(QIcon(flagPath));
        m_langBtn->setText(" " + LM.displayName() + "  ▾");
    }

    // Rebuild avatar menu with fresh translations
    if (m_avatarBtn && m_avatarBtn->menu()) {
        setupMenus();   // re-runs menu build with current language
    }

    if (m_groupSettings) {
        if (QLabel *lbl = m_groupSettings->findChild<QLabel*>("groupTitleLabel"))
            lbl->setText(m_context->translate("menu.settings"));
    }
    if (m_btnSettings) m_btnSettings->setText(m_context->translate("menu.settings"));
    if (m_groupHelp) {
        if (QLabel *lbl = m_groupHelp->findChild<QLabel*>("groupTitleLabel"))
            lbl->setText(m_context->translate("menu.help"));
    }
    if (m_btnAbout) m_btnAbout->setText(m_context->translate("about.title"));
}

void UserAuthPlugin::onLanguageChanged(const QString &/*lang*/) {
    retranslateMenus();
}

// ─── Slots ────────────────────────────────────────────────────────────────────
void UserAuthPlugin::onOpenSettings() {
    UserInfo u = UserManager::instance()->userInfo(m_currentUser);
    bool isAdmin = (u.role == UserRole::Admin);
    SettingsDialog dlg(m_currentUser, isAdmin, m_context, m_context->mainWindow());
    dlg.exec();
}

void UserAuthPlugin::onChangeAvatar() {
    AvatarCropperDialog dlg(m_currentUser, m_context->mainWindow());
    if (dlg.exec() == QDialog::Accepted) {
        setupMenus();
    }
}

void UserAuthPlugin::onLogout() {
    UserManager::instance()->logout();
    m_currentUser.clear();

    if (m_langBtn) {
        m_langBtn->deleteLater();
        m_langBtn = nullptr;
    }
    if (m_avatarBtn) {
        m_avatarBtn->deleteLater();
        m_avatarBtn = nullptr;
    }

    QMenuBar* menuBar = m_context->menuBar();
    if (!menuBar) menuBar = m_context->mainWindow()->menuBar();
    if (menuBar) {
        QWidget* oldCorner = menuBar->cornerWidget(Qt::TopRightCorner);
        if (oldCorner) {
            menuBar->setCornerWidget(nullptr, Qt::TopRightCorner);
            oldCorner->deleteLater();
        }
    }

    m_context->mainWindow()->hide();
    if (!ensureAuthenticated()) {
        qApp->quit();
    } else {
        loadUserSettings();
        checkLicense();
        setupMenus();
        m_context->mainWindow()->show();
    }
}

void UserAuthPlugin::onChangePassword() {
    ChangePasswordDialog dlg(m_currentUser, m_context->mainWindow());
    dlg.exec();
}

void UserAuthPlugin::onAbout() {
    AboutDialog dlg(m_currentUser, m_context->mainWindow());
    dlg.exec();
}
