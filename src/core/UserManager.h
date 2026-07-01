#ifndef USER_MANAGER_H
#define USER_MANAGER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QDate>
#include <QMap>
#include <QSettings>
#include <shared_mutex>
#include "Global.h"

enum class UserRole { Admin, User };

struct APP_EXPORT UserInfo {
    QString username;
    QString passwordHash;
    QString email;
    UserRole role = UserRole::User;
    QString licenseKey;
    QString licenseStatus; // "None", "Trial", "Activated"
    QDate   licenseExpiry;
    QDate   trialStart;
    QString avatarPath;
};

class APP_EXPORT UserManager : public QObject {
    Q_OBJECT
public:
    static UserManager* instance();

    bool loadConfig(const QString &configPath);
    void saveConfig();

    // Auth
    bool login(const QString &username, const QString &password, QString &errorMsg);
    void logout();
    bool isLoggedIn() const;
    UserInfo currentUser() const;
    QString currentUsername() const;

    void setRememberMe(bool enable, const QString &username);
    bool isRememberMeEnabled() const;
    QString savedUsername() const;

    // Password management
    bool changePassword(const QString &username, const QString &oldPass,
                        const QString &newPass, QString &error);
    bool resetPassword(const QString &username, const QString &newPass);
    void updateAvatar(const QString &username, const QString &path);

    // OTP
    bool sendPasswordResetOtp(const QString &email, QString &error);
    bool verifyOtp(const QString &email, const QString &otp);
    QString findUsernameByEmail(const QString &email) const;

    // License
    QString licenseStatusStr(const QString &username) const;
    int     daysRemaining(const QString &username) const;
    bool    needsActivation(const QString &username) const;
    bool    isLicenseExpired(const QString &username) const;
    bool    activateTrial(const QString &username, QString &error);
    bool    activateLicenseKey(const QString &username, const QString &key, QString &error);

    // Admin – users
    QStringList allUsernames() const;
    UserInfo    userInfo(const QString &username) const;
    bool addUser(const UserInfo &info, const QString &plainPassword, QString &error);
    bool updateUser(const UserInfo &info, QString &error);
    bool deleteUser(const QString &username, QString &error);

    // Admin – license keys
    QString     generateLicenseKey();
    QStringList availableLicenseKeys() const;
    bool        revokeLicenseKey(const QString &key, QString &error);

    // SMTP config
    QString smtpSenderEmail() const;
    QString smtpSenderPassword() const;
    void    setSmtpCredentials(const QString &email, const QString &password);

    QString configPath() const { return m_configPath; }

    // Per-user preferences (lưu vào [User_<username>] section của Config_User.ini)
    QString getUserPref(const QString &username, const QString &key, const QString &defaultVal = {}) const;
    void    setUserPref(const QString &username, const QString &key, const QString &value);

private:
    explicit UserManager(QObject *parent = nullptr);

    QString hashPassword(const QString &password) const;
    void    ensureDefaultAdmin();
    void    loadUsers();
    void    saveUsers();
    void    loadLicenseKeys();
    void    saveLicenseKeys();

    mutable std::shared_mutex m_mutex;
    QString m_configPath;
    QSettings *m_settings = nullptr;
    QString    m_currentUsername;

    QMap<QString, UserInfo> m_users;
    QMap<QString, bool>     m_licenseKeys; // key -> isUsed
    QMap<QString, QString>  m_keyOwners;   // key -> username

    struct OtpEntry { QString otp; QDateTime expiry; };
    QMap<QString, OtpEntry> m_otpStore; // email -> entry
};

#endif // USER_MANAGER_H
