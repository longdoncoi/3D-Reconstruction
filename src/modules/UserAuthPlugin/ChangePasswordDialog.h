#ifndef CHANGE_PASSWORD_DIALOG_H
#define CHANGE_PASSWORD_DIALOG_H

#include "../../utils/ModernDialog.h"

class QLineEdit;
class QLabel;
class QPushButton;
class IUserManager;

class ChangePasswordDialog : public ModernDialog {
    Q_OBJECT
public:
    /// `users` is injected by the owner (registry-resolved). See AboutDialog.
    explicit ChangePasswordDialog(const QString &username, IUserManager *users,
                                  QWidget *parent = nullptr);

private slots:
    void onApply();

private:
    void setupUi();
    void applyStyle();
    void setStatus(const QString &msg, bool ok);

    QLineEdit   *m_userEdit = nullptr;
    QLineEdit   *m_oldPassEdit = nullptr;
    QLineEdit   *m_newPassEdit = nullptr;
    QLineEdit   *m_confirmEdit = nullptr;
    QLabel      *m_statusLabel = nullptr;
    QPushButton *m_applyBtn = nullptr;
    QString      m_username;
    bool         m_isExternal = false;
    IUserManager *m_users = nullptr;
};

#endif // CHANGE_PASSWORD_DIALOG_H
