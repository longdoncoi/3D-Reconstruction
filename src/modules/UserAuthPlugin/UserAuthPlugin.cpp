#include "UserAuthPlugin.h"
#include "AppConstants.h"
#include "IAppContext.h"
#include "IShellService.h"
#include "ISettingsService.h"
#include "IUserManager.h"
#include "ILanguageService.h"
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
#include "IAppPaths.h"

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

ILanguageService* UserAuthPlugin::s_lang = nullptr;

QString UserAuthPlugin::translate(const QString& key) {
    return s_lang ? s_lang->translate(key) : key;
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
bool UserAuthPlugin::initialize(IAppContext* context) {
    m_context = context;
    s_lang = context->services()->getRequired<ILanguageService>();
    m_users = context->services()->getRequired<IUserManager>();

    auto* paths = context->services()->getRequired<IAppPaths>();
    QString configDir = QFileInfo(paths ? paths->configPath() : QString()).absolutePath();
    QDir().mkpath(configDir);
    QString configPath = configDir + "/Config_User.ini";
    m_users->loadConfig(configPath);

    if (!ensureAuthenticated()) {
        return false;
    }

    // Load per-user settings (language, theme…) AFTER we know who's logged in
    loadUserSettings();

    checkLicense();
    setupMenus();

    // Connect SignalBus so we retranslate whenever language changes
    connect(m_context->signalBus(), &SignalBus::languageChanged,
            this, &UserAuthPlugin::onLanguageChanged);
    connect(m_context->signalBus(), &SignalBus::agentUiActionRequested, this,
            [this](const QString &action, const QVariantMap &parameters) {
        bool handled = false;
        if (action == "language.change") {
            const QString language = parameters.value("language").toString();
            if (language == "vi" || language == "en") {
                m_context->setLanguage(language);
                saveUserPref("language", language);
                handled = true;
            }
        } else if (action == "admin.settings") {
            onOpenSettings();
            handled = true;
        } else if (action == "admin.change_avatar") {
            onChangeAvatar();
            handled = true;
        } else if (action == "admin.change_password") {
            onChangePassword();
            handled = true;
        } else if (action == "admin.logout") {
            onLogout();
            handled = true;
        } else if (action == "admin.login") {
            QString errorMsg;
            m_users->logout();
            if (m_users->login(parameters.value("username", "Admin").toString(), parameters.value("password", "1").toString(), errorMsg)) {
                m_currentUser = m_users->currentUsername();
                loadUserSettings();
                checkLicense();
                setupMenus();
                handled = true;
            }
        } else if (action == "help.about") {
            onAbout();
            handled = true;
        }
        const QString requestId = parameters.value("request_id").toString();
        const bool ownsAction = action == "language.change" ||
                                 action.startsWith("admin.") ||
                                 action == "help.about";
        // This plugin shares the global action bus.  It must only acknowledge
        // actions it owns; otherwise an unrelated action (for example
        // reconstruction.load_images) is prematurely reported as failed before
        // its actual plugin can send the successful ACK.
        if (!requestId.isEmpty() && ownsAction)
            emit m_context->signalBus()->agentUiActionCompleted(
                requestId, handled, QVariantMap{{"action", action}, {"error", handled ? "" : "Desktop action was not handled"}});
    });
    return true;
}

void UserAuthPlugin::cleanup() {
    // Release everything this plugin installed into the shell FIRST: the
    // host unloads the plugin DLL right after cleanup(), and a widget whose
    // vtable lives in that DLL must not survive it.
    releaseShellWidgets();
    if (m_users) m_users->saveConfig();  // flush while the binding is alive
    m_users = nullptr;
    s_lang = nullptr;
}

void UserAuthPlugin::releaseShellWidgets() {
    // The event filter first: it is the only object of this plugin whose
    // vtable is called by the shell (on every resize of the window).
    releaseOverlayResizer();

    // The menu bar's corner widget owns the two buttons; clear the corner
    // before deleting it so the menu bar does not keep a dangling pointer.
    if (m_context) {
        if (QMenuBar *menuBar = m_context->shell()->menuBar()) {
            if (menuBar->cornerWidget(Qt::TopRightCorner) == m_cornerWidget)
                menuBar->setCornerWidget(nullptr, Qt::TopRightCorner);
        }
    }
    // The buttons are children of the corner widget (the layout re-parents
    // them) or of the menu bar — delete them first, then the empty corner.
    delete m_langBtn;
    m_langBtn = nullptr;
    delete m_avatarBtn;
    m_avatarBtn = nullptr;
    delete m_cornerWidget;
    m_cornerWidget = nullptr;

    delete m_groupHelp;
    m_groupHelp = nullptr;
    delete m_groupSettings;
    m_groupSettings = nullptr;
    m_btnAbout = nullptr;
    m_btnSettings = nullptr;
}

// ─── Authentication ───────────────────────────────────────────────────────────
bool UserAuthPlugin::ensureAuthenticated() {
    auto *um = m_users;
    if (um->isLoggedIn()) {
        m_currentUser = um->currentUsername();
        return true;
    }
    LoginDialog dlg(m_users, m_context->shell()->dialogParent());
    if (dlg.exec() == QDialog::Accepted) {
        m_currentUser = dlg.loggedInUsername();
        return true;
    }
    return false;
}

void UserAuthPlugin::checkLicense() {
    auto *um = m_users;
    if (um->needsActivation(m_currentUser)) {
        LicenseActivationDialog dlg(m_currentUser, m_users, false, m_context->shell()->dialogParent());
        dlg.exec();
    }
    if (um->isLicenseExpired(m_currentUser)) {
        ModernMessageBox::critical(m_context->shell()->dialogParent(),
            translate("common.error"),
            translate("auth.expired"));
        QMetaObject::invokeMethod(qApp, "quit", Qt::QueuedConnection);
    }
}

// ─── Per-user settings ────────────────────────────────────────────────────────
void UserAuthPlugin::loadUserSettings() {
    auto *um = m_users;

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
    applyGlobalBackground(bgPath, bgOpacity, useBg);

    emit m_context->signalBus()->userChanged(m_currentUser);
}

void UserAuthPlugin::applyGlobalBackground(const QString &bgPath, int bgOpacity, bool useBg)
{
    // The plugin is the single owner of the shell's background decoration:
    // the overlay widget AND the event filter that keeps it in sync. The
    // filter's vtable lives in this DLL, so it must be removed before the
    // DLL is unloaded — one owner, one place that does it.
    QWidget *win = m_context ? m_context->shell()->dialogParent() : nullptr;
    OverlayResizer::applyTo(win, bgPath, bgOpacity, useBg, m_overlayResizer);
}

void UserAuthPlugin::releaseOverlayResizer()
{
    if (!m_overlayResizer) return;
    if (QObject *host = m_overlayResizer->parent())
        host->removeEventFilter(m_overlayResizer);
    delete m_overlayResizer;
    m_overlayResizer = nullptr;
}

void UserAuthPlugin::saveUserPref(const QString &key, const QString &value) {
    m_users->setUserPref(m_currentUser, key, value);
}

void UserAuthPlugin::setupMenus() {
    QMenuBar* menuBar = m_context->shell()->menuBar();

    UserInfo u = m_users->userInfo(m_currentUser);

    // ── Inject Help Buttons into tab.help panel ──────────────────────────────
    if (QWidget* panel = m_context->shell()->getTabPanel(AppConstants::TabIds::Help)) {
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
    if (!QFile::exists(flagPath)) flagPath = ":/flags/flag_en.png";
    m_langBtn->setIcon(QIcon(flagPath));
    m_langBtn->setIconSize(QSize(24, 16));
    m_langBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_langBtn->setText(" " + s_lang->displayName() + "  ▾");

    QMenu* langMenu = new QMenu(m_langBtn);
    QAction* viAction = langMenu->addAction(QIcon(":/flags/flag_vn.png"), "Tiếng Việt");
    QAction* enAction = langMenu->addAction(QIcon(":/flags/flag_en.png"), "English");
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

    u = m_users->userInfo(m_currentUser);
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
        // Remember it: the plugin has to release it before the DLL is
        // unloaded (see releaseShellWidgets()).
        m_cornerWidget = cornerWidget;
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
        const QString flagPath = s_lang->currentLanguage() == "vi"
            ? QStringLiteral(":/flags/flag_vn.png")
            : QStringLiteral(":/flags/flag_en.png");
        m_langBtn->setIcon(QIcon(flagPath));
        m_langBtn->setText(" " + s_lang->displayName() + "  ▾");
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
    // The role comes from the core: the core is what enforces it, so asking
    // it is the same answer the guard layer would give.
    const bool isAdmin = m_users->isAdmin();
    SettingsDialog dlg(m_currentUser, isAdmin, m_context, m_users, m_context->shell()->dialogParent());
    // The plugin owns the shell's background decoration, so the dialog asks
    // for a change instead of applying one itself.
    dlg.setBackgroundRequest([this](const QString &path, int opacity, bool enabled) {
        applyGlobalBackground(path, opacity, enabled);
    });
    dlg.exec();
}

void UserAuthPlugin::onChangeAvatar() {
    AvatarCropperDialog dlg(m_currentUser, m_context->services()->getRequired<IAppPaths>(),
                            m_users, m_context->shell()->dialogParent());
    if (dlg.exec() == QDialog::Accepted) {
        setupMenus();
    }
}

void UserAuthPlugin::onLogout() {
    m_users->logout();
    m_currentUser.clear();

    // The menu bar is rebuilt from scratch after the logout: release the
    // widgets of the previous session synchronously (a deleteLater() may not
    // run before the plugin DLL is unloaded).
    releaseShellWidgets();

    m_context->shell()->setVisible(false);
    if (!ensureAuthenticated()) {
        qApp->quit();
    } else {
        loadUserSettings();
        checkLicense();
        setupMenus();
        m_context->shell()->setVisible(true);
    }
}

void UserAuthPlugin::onChangePassword() {
    ChangePasswordDialog dlg(m_currentUser, m_users, m_context->shell()->dialogParent());
    dlg.exec();
}

void UserAuthPlugin::onAbout() {
    AboutDialog dlg(m_currentUser, m_users, m_context->shell()->dialogParent());
    dlg.exec();
}
