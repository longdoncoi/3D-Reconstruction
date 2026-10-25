#ifndef ABOUT_DIALOG_H
#define ABOUT_DIALOG_H

#include "../../utils/ModernDialog.h"

class IUserManager;

class AboutDialog : public ModernDialog {
    Q_OBJECT
public:
    /// `users` is injected by the owner (the plugin resolves it from the
    /// registry). The dialogs never read a process-global binding, so they are
    /// constructible — and testable — without a plugin instance.
    explicit AboutDialog(const QString &username, IUserManager *users,
                         QWidget *parent = nullptr);

private slots:
    void onActivateLicense();

private:
    void setupUi();
    void applyStyle();
    QString m_username;
    IUserManager *m_users = nullptr;
};

#endif // ABOUT_DIALOG_H
