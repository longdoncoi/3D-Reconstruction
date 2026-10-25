#ifndef SETTINGS_DIALOG_H
#define SETTINGS_DIALOG_H

#include "../../utils/BlockingCall.h"
#include "../../utils/ModernDialog.h"
#include "ThemeSelectionDialog.h"
#include <QListWidget>
#include <QStackedWidget>

class IAppContext;
class IUserManager;

class SettingsDialog : public ModernDialog {
    Q_OBJECT
public:
    /// `users` is injected by the owner (registry-resolved). See AboutDialog.
    explicit SettingsDialog(const QString &username, bool isAdmin,
                            IAppContext *ctx, IUserManager *users,
                            QWidget *parent = nullptr);
    ~SettingsDialog() override;

    /// How a background change is applied to the shell. The dialog never
    /// touches the shell window itself — the plugin owns that decoration.
    void setBackgroundRequest(ThemeSelectionDialog::BackgroundRequest request) {
        m_backgroundRequest = std::move(request);
    }

private:
    void setupUi();

    QListWidget    *m_navList   = nullptr;
    QStackedWidget *m_stack     = nullptr;
    QString         m_username;
    bool            m_isAdmin;
    IAppContext    *m_ctx = nullptr;
    IUserManager   *m_users = nullptr;

    ThemeSelectionDialog::BackgroundRequest m_backgroundRequest;

    /// The background tasks of this dialog (the SMTP connection test). The
    /// destructor waits for them: a task must not outlive the service it
    /// calls.
    blocking::TaskTracker m_tasks;
};

#endif // SETTINGS_DIALOG_H
