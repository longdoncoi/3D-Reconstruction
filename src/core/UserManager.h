#ifndef USER_MANAGER_H
#define USER_MANAGER_H

#include "IUserManager.h"
#include "LicenseKeyLedger.h"
#include "OtpStore.h"
#include "UserAccountStore.h"
#include "UserSettingsStore.h"

#include <QObject>
#include <shared_mutex>

#include "Global.h"

/**
 * @brief Concrete singleton implementation of IUserManager.
 *
 * Resolvable both ways:
 *   - legacy: UserManager::instance()
 *   - service-locator: ctx->services()->get<IUserManager>()
 */
class AppShell;
class QSettings;
class TestUserManagerCrypto; // regex test access
class TestMailCredentialLifecycle; // credential lifecycle test access
class TestSmtpOutbox; // SMTP outbox test access

class APP_EXPORT UserManager : public QObject, public IUserManager {
    Q_OBJECT
private:
    static UserManager* instance();
    friend class AppShell;
    friend class TestUserManagerCrypto;
    friend class TestMailCredentialLifecycle;
    friend class TestSmtpOutbox;

    // ── IUserManager ──────────────────────────────────────────────────────────
    bool loadConfig(const QString &configPath) override;
    void saveConfig() override;

    // Auth
    bool login(const QString &usernameOrEmail, const QString &password, QString &errorMsg,
               AuthError *code = nullptr) override;
    void logout() override;
    bool isLoggedIn() const override;
    UserInfo currentUser() const override;
    QString currentUsername() const override;

    void setRememberMe(bool enable, const QString &username) override;
    bool isRememberMeEnabled() const override;
    QString savedUsername() const override;

    // Password management
    bool changePassword(const QString &username, const QString &oldPass,
                        const QString &newPass, QString &error,
                        AuthError *code = nullptr) override;
    bool resetPassword(const QString &username, const QString &newPass) override;
    void updateAvatar(const QString &username, const QString &path) override;

    // OTP
    bool sendPasswordResetOtp(const QString &email, QString &error) override;
    bool verifyOtp(const QString &email, const QString &otp) override;
    QString findUsernameByEmail(const QString &email) const override;

    // License
    QString licenseStatusStr(const QString &username) const override;
    int     daysRemaining(const QString &username) const override;
    bool    needsActivation(const QString &username) const override;
    bool    isLicenseExpired(const QString &username) const override;
    bool    activateTrial(const QString &username, QString &error,
                          AuthError *code = nullptr) override;
    bool    activateLicenseKey(const QString &username, const QString &key, QString &error,
                               AuthError *code = nullptr) override;

    // Admin – users
    QStringList allUsernames() const override;
    UserInfo    userInfo(const QString &username) const override;
    bool addUser(const UserInfo &info, const QString &plainPassword, QString &error,
                 AuthError *code = nullptr) override;
    bool updateUser(const UserInfo &info, QString &error) override;
    bool deleteUser(const QString &username, QString &error) override;

    // Admin – license keys
    QString     generateLicenseKey() override;
    QStringList availableLicenseKeys() const override;
    bool        revokeLicenseKey(const QString &key, QString &error) override;

    // SMTP config
    QString smtpSenderEmail() const override;
    QString smtpSenderPassword() const override;
    void    setSmtpCredentials(const QString &email, const QString &password) override;

    QString configPath() const override { return m_configPath; }

    // ── Outgoing mail seam ───────────────────────────────────────────────────
    /**
     * Injected by the composition root (AppShell) so core code never touches
     * the concrete SMTP transport in services/mail. Not part of IUserManager:
     * this is a wiring concern, not an application contract.
     */
    void setOutgoingMail(class IOutgoingMail *outbox) { m_outgoingMail = outbox; }

    // Per-user preferences (lưu vào [User_<username>] section của Config_User.ini)
    QString getUserPref(const QString &username, const QString &key, const QString &defaultVal = {}) const override;
    void    setUserPref(const QString &username, const QString &key, const QString &value) override;

    // Per-user credentials — stored through SecretVault, never as plaintext.
    QString getUserSecret(const QString &username, const QString &key, const QString &defaultVal = {}) const override;
    void    setUserSecret(const QString &username, const QString &key, const QString &value) override;

private:
    explicit UserManager(QObject *parent = nullptr);

    mutable std::shared_mutex m_mutex;
    QString m_configPath;
    QSettings *m_settings = nullptr;
    QString    m_currentUsername;

    UserAccountStore  m_accounts;
    LicenseKeyLedger  m_licenseLedger;
    OtpStore          m_otpStore;
    UserSettingsStore m_userSettings;

    IOutgoingMail *m_outgoingMail = nullptr; ///< injected, not owned
};

#endif // USER_MANAGER_H
