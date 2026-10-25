#ifndef USER_AUTH_PLUGIN_H
#define USER_AUTH_PLUGIN_H

#include <QObject>
#include "IPlugin.h"
#include "AppConstants.h"
#include "OverlayResizer.h"

class IAppContext;
class ILanguageService;
class IUserManager;
class QToolButton;
class QGroupBox;
class QWidget;

class UserAuthPlugin : public QObject, public IPlugin {
    Q_OBJECT
    Q_PLUGIN_METADATA(IID IPlugin_iid)
    Q_INTERFACES(IPlugin)

public:
    static QString translate(const QString& key);
    /** Module-scoped ILanguageService binding (registry-sourced), for dialogs
     *  that need the service itself (e.g. ModernMessageBox::question). */
    static ILanguageService* languageService() { return s_lang; }

    QString pluginName() const override
    {
        return QString::fromLatin1(AppConstants::PluginNames::UserAuthentication);
    }
    bool initialize(IAppContext* context) override;
    void cleanup() override;
    int  loadOrder() const override { return 5; } // Load rất sớm, trước các module khác

private slots:
    void onLogout();
    void onChangePassword();
    void onAbout();
    void onChangeAvatar();
    void onLanguageChanged(const QString &lang);
    void onOpenSettings();

private:
    void setupMenus();
    void retranslateMenus();
    bool ensureAuthenticated();
    void checkLicense();
    void loadUserSettings();
    void saveUserPref(const QString &key, const QString &value);

    /// Remove every widget this plugin installed into the shell.
    ///
    /// The delete is SYNCHRONOUS on purpose: a deleteLater() may not run
    /// before the plugin DLL is unloaded, and a widget whose vtable lives in
    /// that DLL must not survive it. Called by cleanup() and by the logout
    /// path (which rebuilds the menu bar from scratch).
    void releaseShellWidgets();

    /// Apply (or remove) the background image on the shell window.
    ///
    /// The plugin is the single owner of the overlay and of its event
    /// filter: the filter's vtable lives in this DLL, so it has to be
    /// removed before the unload, and a second owner would be a second
    /// chance to forget it.
    void applyGlobalBackground(const QString &bgPath, int bgOpacity, bool useBg);

    /// Remove the overlay's event filter from the shell and delete it.
    void releaseOverlayResizer();

    IAppContext *m_context = nullptr;
    QString m_currentUser;

    /// The user service, resolved from the registry during initialize() and
    /// cleared on cleanup(). Dialogs receive it through their constructor —
    /// there is no process-global binding, so a dialog is constructible
    /// without a plugin instance (and two plugin instances cannot share it).
    IUserManager *m_users = nullptr;

    /// Module-scoped translation source, bound to the registry-provided
    /// ILanguageService during initialize() (cleared on cleanup). Feeds the
    /// static translate() facade used across this module's dialogs.
    static ILanguageService *s_lang;

    // UI elements that need retranslation
    QToolButton *m_avatarBtn    = nullptr;
    QToolButton *m_langBtn      = nullptr;

    QGroupBox   *m_groupSettings = nullptr;
    QGroupBox   *m_groupHelp     = nullptr;
    QToolButton *m_btnSettings   = nullptr;   // single "Cài đặt" ribbon button
    QToolButton *m_btnAbout      = nullptr;

    /// The menu bar's corner widget holding the language and avatar buttons.
    QWidget     *m_cornerWidget  = nullptr;

    /// The event filter that keeps the background overlay in sync with the
    /// shell window. It lives in the shell's widget tree while its vtable
    /// lives in this DLL, so it has to be removed before the unload.
    /// nullptr means "no filter installed" — the handle IS the state.
    OverlayResizer *m_overlayResizer = nullptr;
};

#endif // USER_AUTH_PLUGIN_H
