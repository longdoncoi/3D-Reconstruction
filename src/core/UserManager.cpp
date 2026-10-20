#include "UserManager.h"

#include "CryptoUtils.h"
#include "IOutgoingMail.h"

#include <QDate>
#include <QSettings>

UserManager *UserManager::instance() {
    static UserManager instance;
    return &instance;
}

UserManager::UserManager(QObject *parent) : QObject(parent) {}

bool UserManager::isLoggedIn() const { std::shared_lock lock(m_mutex); return !m_currentUsername.isEmpty(); }
UserInfo UserManager::currentUser() const { std::shared_lock lock(m_mutex); return m_accounts.info(m_currentUsername); }
QString UserManager::currentUsername() const { std::shared_lock lock(m_mutex); return m_currentUsername; }

QStringList UserManager::allUsernames() const { std::shared_lock lock(m_mutex); return m_accounts.usernames(); }
UserInfo UserManager::userInfo(const QString &username) const { std::shared_lock lock(m_mutex); return m_accounts.info(username); }

// ─── Config ──────────────────────────────────────────────────────────────────

bool UserManager::loadConfig(const QString &configPath) {
    std::unique_lock lock(m_mutex);
    m_configPath = configPath;
    if (m_settings) {
        delete m_settings;
    }
    m_settings = new QSettings(configPath, QSettings::IniFormat, this);
    m_accounts.attachSettings(m_settings);
    m_licenseLedger.attachSettings(m_settings);
    m_userSettings.attachSettings(m_settings);
    m_accounts.load();
    m_licenseLedger.load();
    m_accounts.ensureDefaultAdmin(m_configPath);
    if (m_settings) {
        m_accounts.save();
        m_licenseLedger.save();
        m_settings->sync();
    }
    return true;
}

void UserManager::saveConfig() {
    std::unique_lock lock(m_mutex);
    if (!m_settings) return;
    m_accounts.save();
    m_licenseLedger.save();
    m_settings->sync();
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

bool UserManager::login(const QString &usernameOrEmail, const QString &password,
                        QString &errorMsg, AuthError *code) {
    std::unique_lock lock(m_mutex);
    const QString foundUsername = m_accounts.findLoginUsername(usernameOrEmail);
    if (foundUsername.isEmpty()) {
        errorMsg = "Tên đăng nhập hoặc Email không tồn tại.";
        if (code) *code = AuthError::UserNotFound;
        return false;
    }
    const QString storedHash = m_accounts.passwordHash(foundUsername);
    if (!CryptoUtils::verifyPassword(password, storedHash)) {
        errorMsg = "Mật khẩu không đúng.";
        if (code) *code = AuthError::WrongPassword;
        return false;
    }
    // Transparent upgrade: rehash legacy hashes to the PBKDF2 format once the
    // plaintext password has been proven correct.
    if (!CryptoUtils::isModernPasswordHash(storedHash)) {
        m_accounts.setPasswordHash(foundUsername, CryptoUtils::hashPassword(password));
        if (m_settings) { m_accounts.save(); m_settings->sync(); }
    }
    m_currentUsername = foundUsername;
    if (m_settings) {
        m_settings->setValue("Session/last_user", foundUsername);
        m_settings->sync();
    }
    return true;
}

void UserManager::setRememberMe(bool enable, const QString &username) {
    std::unique_lock lock(m_mutex);
    if (!m_settings) return;
    m_settings->setValue("Session/remember_me", enable);
    if (enable) {
        m_settings->setValue("Session/saved_username", username);
    } else {
        m_settings->remove("Session/saved_username");
    }
    m_settings->sync();
}

bool UserManager::isRememberMeEnabled() const {
    std::shared_lock lock(m_mutex);
    return m_settings ? m_settings->value("Session/remember_me", false).toBool() : false;
}

QString UserManager::savedUsername() const {
    std::shared_lock lock(m_mutex);
    return m_settings ? m_settings->value("Session/saved_username", "").toString() : QString();
}

void UserManager::logout() {
    std::unique_lock lock(m_mutex);
    m_currentUsername.clear();
}

// ─── Password ─────────────────────────────────────────────────────────────────

bool UserManager::changePassword(const QString &username, const QString &oldPass,
                                  const QString &newPass, QString &error,
                                  AuthError *code) {
    std::unique_lock lock(m_mutex);
    if (!m_accounts.contains(username)) { error = "User không tồn tại."; if (code) *code = AuthError::UserNotFound; return false; }
    if (!CryptoUtils::verifyPassword(oldPass, m_accounts.passwordHash(username))) {
        error = "Mật khẩu cũ không đúng.";
        if (code) *code = AuthError::WrongPassword;
        return false;
    }
    if (newPass.length() < 6) { error = "Mật khẩu mới phải có ít nhất 6 ký tự."; if (code) *code = AuthError::WeakPassword; return false; }
    m_accounts.setPasswordHash(username, CryptoUtils::hashPassword(newPass));
    if (m_settings) { m_accounts.save(); m_settings->sync(); }
    return true;
}

bool UserManager::resetPassword(const QString &username, const QString &newPass) {
    std::unique_lock lock(m_mutex);
    if (!m_accounts.contains(username)) return false;
    m_accounts.setPasswordHash(username, CryptoUtils::hashPassword(newPass));
    if (m_settings) { m_accounts.save(); m_settings->sync(); }
    return true;
}

void UserManager::updateAvatar(const QString &username, const QString &path) {
    std::unique_lock lock(m_mutex);
    if (m_accounts.contains(username)) {
        UserInfo user = m_accounts.info(username);
        user.avatarPath = path;
        m_accounts.update(user);
        if (m_settings) { m_accounts.save(); m_settings->sync(); }
    }
}

// ─── OTP ──────────────────────────────────────────────────────────────────────

QString UserManager::findUsernameByEmail(const QString &email) const {
    std::shared_lock lock(m_mutex);
    return m_accounts.findUsernameByEmail(email);
}

bool UserManager::sendPasswordResetOtp(const QString &email, QString &error) {
    if (findUsernameByEmail(email).isEmpty()) {
        error = "Email không được đăng ký trong hệ thống.";
        return false;
    }
    const QString otp = m_otpStore.issue(email);

    if (!m_outgoingMail) {
        error = "Dịch vụ gửi email chưa khả dụng.";
        return false;
    }

    QString senderEmail;
    {
        std::shared_lock lock(m_mutex);
        senderEmail = m_userSettings.smtpSenderEmail();
    }
    if (senderEmail.isEmpty()) {
        error = "Chưa cấu hình email gửi. Vui lòng liên hệ Admin.";
        return false;
    }

    QString subject = "[3D-Reconstruction] Mã xác nhận đặt lại mật khẩu";
    QString body    = QString("Mã xác nhận của bạn là: <b>%1</b><br><br>"
                              "Mã có hiệu lực trong 10 phút.<br>"
                              "Nếu bạn không yêu cầu đặt lại mật khẩu, hãy bỏ qua email này.").arg(otp);
    return m_outgoingMail->send(senderEmail, email, subject, body, error);
}

bool UserManager::verifyOtp(const QString &email, const QString &otp) {
    return m_otpStore.verify(email, otp);
}

// ─── License ──────────────────────────────────────────────────────────────────

bool UserManager::needsActivation(const QString &username) const {
    std::shared_lock lock(m_mutex);
    if (!m_accounts.contains(username)) return false;
    return m_accounts.info(username).licenseStatus == "None";
}

bool UserManager::isLicenseExpired(const QString &username) const {
    std::shared_lock lock(m_mutex);
    if (!m_accounts.contains(username)) return true;
    const UserInfo user = m_accounts.info(username);
    if (user.licenseStatus == "None") return true;
    if (!user.licenseExpiry.isValid()) return false;
    return QDate::currentDate() > user.licenseExpiry;
}

QString UserManager::licenseStatusStr(const QString &username) const {
    std::shared_lock lock(m_mutex);
    return m_accounts.info(username).licenseStatus;
}

int UserManager::daysRemaining(const QString &username) const {
    std::shared_lock lock(m_mutex);
    if (!m_accounts.contains(username)) return 0;
    const UserInfo user = m_accounts.info(username);
    if (!user.licenseExpiry.isValid()) return 0;
    int days = QDate::currentDate().daysTo(user.licenseExpiry);
    return qMax(0, (int)days);
}

bool UserManager::activateTrial(const QString &username, QString &error,
                                 AuthError *code) {
    std::unique_lock lock(m_mutex);
    if (!m_accounts.contains(username)) { error = "User không tồn tại."; if (code) *code = AuthError::UserNotFound; return false; }
    UserInfo user = m_accounts.info(username);
    if (user.licenseStatus != "None") { error = "Tài khoản đã được kích hoạt."; if (code) *code = AuthError::AlreadyActivated; return false; }
    user.licenseStatus = "Trial";
    user.trialStart    = QDate::currentDate();
    user.licenseExpiry = QDate::currentDate().addDays(30);
    m_accounts.update(user);
    if (m_settings) { m_accounts.save(); m_settings->sync(); }
    return true;
}

bool UserManager::activateLicenseKey(const QString &username, const QString &key,
                                     QString &error, AuthError *code) {
    std::unique_lock lock(m_mutex);
    if (!m_accounts.contains(username)) { error = "User không tồn tại."; if (code) *code = AuthError::UserNotFound; return false; }
    if (!m_licenseLedger.contains(key)) { error = "License key không hợp lệ."; if (code) *code = AuthError::InvalidKey; return false; }
    if (m_licenseLedger.isUsed(key)) { error = "License key đã được sử dụng."; if (code) *code = AuthError::KeyAlreadyUsed; return false; }
    UserInfo user = m_accounts.info(username);
    user.licenseKey    = key;
    user.licenseStatus = "Activated";
    user.licenseExpiry = QDate::currentDate().addYears(1);
    m_accounts.update(user);
    m_licenseLedger.claim(key, username);
    if (m_settings) {
        m_accounts.save();
        m_licenseLedger.save();
        m_settings->sync();
    }
    return true;
}

// ─── Admin – Users ────────────────────────────────────────────────────────────

bool UserManager::addUser(const UserInfo &info, const QString &plainPassword,
                          QString &error, AuthError *code) {
    std::unique_lock lock(m_mutex);
    if (m_accounts.contains(info.username)) { error = "Tên đăng nhập đã tồn tại."; if (code) *code = AuthError::UserExists; return false; }
    if (plainPassword.length() < 6) { error = "Mật khẩu phải có ít nhất 6 ký tự."; if (code) *code = AuthError::WeakPassword; return false; }
    UserInfo user = info;
    user.licenseStatus = "None";
    m_accounts.insert(user, CryptoUtils::hashPassword(plainPassword));
    if (m_settings) { m_accounts.save(); m_settings->sync(); }
    return true;
}

bool UserManager::updateUser(const UserInfo &info, QString &error) {
    std::unique_lock lock(m_mutex);
    if (!m_accounts.contains(info.username)) { error = "User không tồn tại."; return false; }
    // The password hash lives in UserAccountStore and is intentionally not
    // part of UserInfo, so a profile update cannot clobber the credential.
    m_accounts.update(info);
    if (m_settings) { m_accounts.save(); m_settings->sync(); }
    return true;
}

bool UserManager::deleteUser(const QString &username, QString &error) {
    std::unique_lock lock(m_mutex);
    if (username == "admin") { error = "Không thể xóa tài khoản admin mặc định."; return false; }
    if (username == m_currentUsername) { error = "Không thể xóa tài khoản đang đăng nhập."; return false; }
    if (!m_accounts.contains(username)) { error = "User không tồn tại."; return false; }
    m_accounts.remove(username);
    if (m_settings) {
        m_accounts.save();
        m_settings->sync();
    }
    return true;
}

// ─── Admin – License Keys ─────────────────────────────────────────────────────

QString UserManager::generateLicenseKey() {
    std::unique_lock lock(m_mutex);
    const QString key = m_licenseLedger.generate();
    if (m_settings) {
        m_licenseLedger.save();
        m_settings->sync();
    }
    return key;
}

QStringList UserManager::availableLicenseKeys() const {
    std::shared_lock lock(m_mutex);
    return m_licenseLedger.available();
}

bool UserManager::revokeLicenseKey(const QString &key, QString &error) {
    std::unique_lock lock(m_mutex);
    QString previousOwner;
    if (!m_licenseLedger.release(key, previousOwner)) {
        error = "Key không tồn tại.";
        return false;
    }
    // Revoke from the user that consumed the key, if that account still exists.
    if (!previousOwner.isEmpty() && m_accounts.contains(previousOwner)) {
        UserInfo user = m_accounts.info(previousOwner);
        user.licenseStatus = "None";
        user.licenseKey.clear();
        user.licenseExpiry = QDate();
        m_accounts.update(user);
    }
    if (m_settings) {
        m_accounts.save();
        m_licenseLedger.save();
        m_settings->sync();
    }
    return true;
}

// ─── SMTP Config ──────────────────────────────────────────────────────────────

QString UserManager::smtpSenderEmail() const {
    std::shared_lock lock(m_mutex);
    return m_userSettings.smtpSenderEmail();
}

QString UserManager::smtpSenderPassword() const {
    std::shared_lock lock(m_mutex);
    return m_userSettings.smtpSenderPassword();
}

void UserManager::setSmtpCredentials(const QString &email, const QString &password) {
    std::unique_lock lock(m_mutex);
    m_userSettings.setSmtpCredentials(email, password);
}

// ─── Per-user Preferences ─────────────────────────────────────────────────────

QString UserManager::getUserSecret(const QString &username, const QString &key, const QString &defaultVal) const {
    std::shared_lock lock(m_mutex);
    return m_userSettings.getUserSecret(username, key, defaultVal);
}

void UserManager::setUserSecret(const QString &username, const QString &key, const QString &value) {
    std::unique_lock lock(m_mutex);
    m_userSettings.setUserSecret(username, key, value);
}

// ─── Per-user Preferences ─────────────────────────────────────────────────────

QString UserManager::getUserPref(const QString &username, const QString &key, const QString &defaultVal) const {
    std::shared_lock lock(m_mutex);
    return m_userSettings.getUserPref(username, key, defaultVal);
}

void UserManager::setUserPref(const QString &username, const QString &key, const QString &value) {
    std::unique_lock lock(m_mutex);
    m_userSettings.setUserPref(username, key, value);
}
