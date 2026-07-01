#include "UserManager.h"
#include "SmtpMailer.h"

#include <QCryptographicHash>
#include <QRandomGenerator>
#include <QDebug>

UserManager *UserManager::instance() {
    static UserManager* instance = new UserManager();
    return instance;
}

UserManager::UserManager(QObject *parent) : QObject(parent) {}

bool UserManager::isLoggedIn() const { std::shared_lock lock(m_mutex); return !m_currentUsername.isEmpty(); }
UserInfo UserManager::currentUser() const { std::shared_lock lock(m_mutex); return m_users.value(m_currentUsername); }
QString UserManager::currentUsername() const { std::shared_lock lock(m_mutex); return m_currentUsername; }

QStringList UserManager::allUsernames() const { std::shared_lock lock(m_mutex); return m_users.keys(); }
UserInfo UserManager::userInfo(const QString &username) const { std::shared_lock lock(m_mutex); return m_users.value(username); }

// ─── Config ──────────────────────────────────────────────────────────────────

bool UserManager::loadConfig(const QString &configPath) {
    std::unique_lock lock(m_mutex);
    m_configPath = configPath;
    if (m_settings) {
        delete m_settings;
    }
    m_settings = new QSettings(configPath, QSettings::IniFormat, this);
    loadUsers();
    loadLicenseKeys();
    ensureDefaultAdmin();
    if (m_settings) {
        saveUsers();
        saveLicenseKeys();
        m_settings->sync();
    }
    return true;
}

void UserManager::saveConfig() {
    std::unique_lock lock(m_mutex);
    if (!m_settings) return;
    saveUsers();
    saveLicenseKeys();
    m_settings->sync();
}

void UserManager::loadUsers() {
    m_users.clear();
    int count = m_settings->value("Users/count", 0).toInt();
    QStringList names = m_settings->value("Users/usernames").toStringList();
    for (const QString &name : names) {
        QString section = "User_" + name;
        UserInfo u;
        u.username     = name;
        u.passwordHash = m_settings->value(section + "/password_hash").toString();
        u.email        = m_settings->value(section + "/email").toString();
        u.role         = m_settings->value(section + "/role", "User").toString() == "Admin"
                             ? UserRole::Admin : UserRole::User;
        u.licenseKey    = m_settings->value(section + "/license_key").toString();
        u.licenseStatus = m_settings->value(section + "/license_status", "None").toString();
        QString expiry  = m_settings->value(section + "/license_expiry").toString();
        QString trial   = m_settings->value(section + "/trial_start").toString();
        if (!expiry.isEmpty()) u.licenseExpiry = QDate::fromString(expiry, Qt::ISODate);
        if (!trial.isEmpty())  u.trialStart    = QDate::fromString(trial, Qt::ISODate);
        u.avatarPath = m_settings->value(section + "/avatar_path").toString();
        m_users.insert(name, u);
    }
    Q_UNUSED(count)
}

void UserManager::saveUsers() {
    QStringList names = m_users.keys();
    m_settings->setValue("Users/count", names.size());
    m_settings->setValue("Users/usernames", names);
    for (const QString &name : names) {
        const UserInfo &u = m_users[name];
        QString section = "User_" + name;
        m_settings->setValue(section + "/password_hash", u.passwordHash);
        m_settings->setValue(section + "/email",         u.email);
        m_settings->setValue(section + "/role",          u.role == UserRole::Admin ? "Admin" : "User");
        m_settings->setValue(section + "/license_key",    u.licenseKey);
        m_settings->setValue(section + "/license_status", u.licenseStatus);
        m_settings->setValue(section + "/license_expiry",
                             u.licenseExpiry.isValid() ? u.licenseExpiry.toString(Qt::ISODate) : "");
        m_settings->setValue(section + "/trial_start",
                             u.trialStart.isValid() ? u.trialStart.toString(Qt::ISODate) : "");
        m_settings->setValue(section + "/avatar_path", u.avatarPath);
    }
}

void UserManager::loadLicenseKeys() {
    m_licenseKeys.clear();
    m_keyOwners.clear();
    QStringList keys = m_settings->value("LicenseKeys/keys").toStringList();
    for (const QString &k : keys) {
        bool used = m_settings->value("LicenseKeys/" + k + "_used", false).toBool();
        QString owner = m_settings->value("LicenseKeys/" + k + "_user").toString();
        m_licenseKeys[k] = used;
        if (!owner.isEmpty()) m_keyOwners[k] = owner;
    }
}

void UserManager::saveLicenseKeys() {
    QStringList keys = m_licenseKeys.keys();
    m_settings->setValue("LicenseKeys/keys", keys);
    for (const QString &k : keys) {
        m_settings->setValue("LicenseKeys/" + k + "_used", m_licenseKeys[k]);
        m_settings->setValue("LicenseKeys/" + k + "_user", m_keyOwners.value(k));
    }
}

void UserManager::ensureDefaultAdmin() {
    if (!m_users.contains("admin")) {
        UserInfo admin;
        admin.username     = "admin";
        admin.passwordHash = hashPassword("1");
        admin.email        = "admin@example.com";
        admin.role         = UserRole::Admin;
        admin.licenseStatus = "Activated";
        admin.licenseExpiry = QDate::currentDate().addYears(10);
        m_users["admin"] = admin;
        qDebug() << "[UserAuth] Default admin created. Password: 1";
    }
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

bool UserManager::login(const QString &usernameOrEmail, const QString &password, QString &errorMsg) {
    std::unique_lock lock(m_mutex);
    QString foundUsername;
    QString lowerInput = usernameOrEmail.toLower();

    for (auto it = m_users.begin(); it != m_users.end(); ++it) {
        if (it.value().username.toLower() == lowerInput || it.value().email.toLower() == lowerInput) {
            foundUsername = it.key();
            break;
        }
    }

    if (foundUsername.isEmpty()) {
        errorMsg = "Tên đăng nhập hoặc Email không tồn tại.";
        return false;
    }
    if (m_users[foundUsername].passwordHash != hashPassword(password)) {
        errorMsg = "Mật khẩu không đúng.";
        return false;
    }
    m_currentUsername = foundUsername;
    m_settings->setValue("Session/last_user", foundUsername);
    m_settings->sync();
    return true;
}

void UserManager::setRememberMe(bool enable, const QString &username) {
    std::unique_lock lock(m_mutex);
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
    return m_settings->value("Session/remember_me", false).toBool();
}

QString UserManager::savedUsername() const {
    std::shared_lock lock(m_mutex);
    return m_settings->value("Session/saved_username", "").toString();
}

void UserManager::logout() {
    std::unique_lock lock(m_mutex);
    m_currentUsername.clear();
}

// ─── Password ─────────────────────────────────────────────────────────────────

QString UserManager::hashPassword(const QString &password) const {
    return QString::fromLatin1(
        QCryptographicHash::hash(password.toUtf8(), QCryptographicHash::Sha256).toHex());
}

bool UserManager::changePassword(const QString &username, const QString &oldPass,
                                  const QString &newPass, QString &error) {
    std::unique_lock lock(m_mutex);
    if (!m_users.contains(username)) { error = "User không tồn tại."; return false; }
    if (m_users[username].passwordHash != hashPassword(oldPass)) {
        error = "Mật khẩu cũ không đúng."; return false;
    }
    if (newPass.length() < 6) { error = "Mật khẩu mới phải có ít nhất 6 ký tự."; return false; }
    m_users[username].passwordHash = hashPassword(newPass);
    if (m_settings) { saveUsers(); m_settings->sync(); }
    return true;
}

bool UserManager::resetPassword(const QString &username, const QString &newPass) {
    std::unique_lock lock(m_mutex);
    if (!m_users.contains(username)) return false;
    m_users[username].passwordHash = hashPassword(newPass);
    if (m_settings) { saveUsers(); m_settings->sync(); }
    return true;
}

void UserManager::updateAvatar(const QString &username, const QString &path) {
    std::unique_lock lock(m_mutex);
    if (m_users.contains(username)) {
        m_users[username].avatarPath = path;
        if (m_settings) { saveUsers(); m_settings->sync(); }
    }
}

// ─── OTP ──────────────────────────────────────────────────────────────────────

QString UserManager::findUsernameByEmail(const QString &email) const {
    std::shared_lock lock(m_mutex);
    for (auto it = m_users.begin(); it != m_users.end(); ++it)
        if (it.value().email.toLower() == email.toLower())
            return it.key();
    return {};
}

bool UserManager::sendPasswordResetOtp(const QString &email, QString &error) {
    if (findUsernameByEmail(email).isEmpty()) {
        error = "Email không được đăng ký trong hệ thống.";
        return false;
    }
    // Generate 6-digit OTP
    int code = QRandomGenerator::global()->bounded(100000, 999999);
    QString otp = QString::number(code);
    {
        std::unique_lock lock(m_mutex);
        m_otpStore[email.toLower()] = { otp, QDateTime::currentDateTime().addSecs(600) };
    }

    QString senderEmail = smtpSenderEmail();
    QString senderPass  = smtpSenderPassword();
    if (senderEmail.isEmpty()) {
        error = "Chưa cấu hình email gửi. Vui lòng liên hệ Admin.";
        return false;
    }

    SmtpMailer mailer(senderEmail, senderPass);
    QString subject = "[3D-Reconstruction] Mã xác nhận đặt lại mật khẩu";
    QString body    = QString("Mã xác nhận của bạn là: <b>%1</b><br><br>"
                              "Mã có hiệu lực trong 10 phút.<br>"
                              "Nếu bạn không yêu cầu đặt lại mật khẩu, hãy bỏ qua email này.").arg(otp);
    return mailer.sendMail(senderEmail, email, subject, body, error);
}

bool UserManager::verifyOtp(const QString &email, const QString &otp) {
    std::unique_lock lock(m_mutex);
    QString key = email.toLower();
    if (!m_otpStore.contains(key)) return false;
    const auto &entry = m_otpStore[key];
    if (QDateTime::currentDateTime() > entry.expiry) {
        m_otpStore.remove(key);
        return false;
    }
    if (entry.otp != otp) return false;
    m_otpStore.remove(key);
    return true;
}

// ─── License ──────────────────────────────────────────────────────────────────

bool UserManager::needsActivation(const QString &username) const {
    std::shared_lock lock(m_mutex);
    if (!m_users.contains(username)) return false;
    const auto &u = m_users[username];
    return u.licenseStatus == "None";
}

bool UserManager::isLicenseExpired(const QString &username) const {
    std::shared_lock lock(m_mutex);
    if (!m_users.contains(username)) return true;
    const auto &u = m_users[username];
    if (u.licenseStatus == "None") return true;
    if (!u.licenseExpiry.isValid()) return false;
    return QDate::currentDate() > u.licenseExpiry;
}

QString UserManager::licenseStatusStr(const QString &username) const {
    std::shared_lock lock(m_mutex);
    return m_users.value(username).licenseStatus;
}

int UserManager::daysRemaining(const QString &username) const {
    std::shared_lock lock(m_mutex);
    if (!m_users.contains(username)) return 0;
    const auto &u = m_users[username];
    if (!u.licenseExpiry.isValid()) return 0;
    int days = QDate::currentDate().daysTo(u.licenseExpiry);
    return qMax(0, (int)days);
}

bool UserManager::activateTrial(const QString &username, QString &error) {
    std::unique_lock lock(m_mutex);
    if (!m_users.contains(username)) { error = "User không tồn tại."; return false; }
    auto &u = m_users[username];
    if (u.licenseStatus != "None") { error = "Tài khoản đã được kích hoạt."; return false; }
    u.licenseStatus = "Trial";
    u.trialStart    = QDate::currentDate();
    u.licenseExpiry = QDate::currentDate().addDays(30);
    if (m_settings) { saveUsers(); m_settings->sync(); }
    return true;
}

bool UserManager::activateLicenseKey(const QString &username, const QString &key, QString &error) {
    std::unique_lock lock(m_mutex);
    if (!m_users.contains(username)) { error = "User không tồn tại."; return false; }
    if (!m_licenseKeys.contains(key)) { error = "License key không hợp lệ."; return false; }
    if (m_licenseKeys[key]) { error = "License key đã được sử dụng."; return false; }
    auto &u = m_users[username];
    u.licenseKey    = key;
    u.licenseStatus = "Activated";
    u.licenseExpiry = QDate::currentDate().addYears(1);
    m_licenseKeys[key] = true;
    m_keyOwners[key]   = username;
    if (m_settings) {
        saveUsers();
        saveLicenseKeys();
        m_settings->sync();
    }
    return true;
}

// ─── Admin – Users ────────────────────────────────────────────────────────────

bool UserManager::addUser(const UserInfo &info, const QString &plainPassword, QString &error) {
    std::unique_lock lock(m_mutex);
    if (m_users.contains(info.username)) { error = "Tên đăng nhập đã tồn tại."; return false; }
    if (plainPassword.length() < 6) { error = "Mật khẩu phải có ít nhất 6 ký tự."; return false; }
    UserInfo u = info;
    u.passwordHash = hashPassword(plainPassword);
    u.licenseStatus = "None";
    m_users[info.username] = u;
    if (m_settings) { saveUsers(); m_settings->sync(); }
    return true;
}

bool UserManager::updateUser(const UserInfo &info, QString &error) {
    std::unique_lock lock(m_mutex);
    if (!m_users.contains(info.username)) { error = "User không tồn tại."; return false; }
    // Preserve password hash
    UserInfo u = info;
    u.passwordHash = m_users[info.username].passwordHash;
    m_users[info.username] = u;
    if (m_settings) { saveUsers(); m_settings->sync(); }
    return true;
}

bool UserManager::deleteUser(const QString &username, QString &error) {
    std::unique_lock lock(m_mutex);
    if (username == "admin") { error = "Không thể xóa tài khoản admin mặc định."; return false; }
    if (username == m_currentUsername) { error = "Không thể xóa tài khoản đang đăng nhập."; return false; }
    if (!m_users.contains(username)) { error = "User không tồn tại."; return false; }
    m_users.remove(username);
    // Remove user section from INI
    if (m_settings) {
        m_settings->remove("User_" + username);
        saveUsers();
        m_settings->sync();
    }
    return true;
}

// ─── Admin – License Keys ─────────────────────────────────────────────────────

QString UserManager::generateLicenseKey() {
    const QString chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // exclude ambiguous 0,1,I,O
    QString key;
    for (int g = 0; g < 4; g++) {
        for (int i = 0; i < 5; i++)
            key += chars[QRandomGenerator::global()->bounded(chars.size())];
        if (g < 3) key += '-';
    }
    std::unique_lock lock(m_mutex);
    m_licenseKeys[key] = false;
    if (m_settings) {
        saveLicenseKeys();
        m_settings->sync();
    }
    return key;
}

QStringList UserManager::availableLicenseKeys() const {
    std::shared_lock lock(m_mutex);
    QStringList result;
    for (auto it = m_licenseKeys.begin(); it != m_licenseKeys.end(); ++it)
        result << it.key();
    return result;
}

bool UserManager::revokeLicenseKey(const QString &key, QString &error) {
    std::unique_lock lock(m_mutex);
    if (!m_licenseKeys.contains(key)) { error = "Key không tồn tại."; return false; }
    // Revoke from user if assigned
    if (m_keyOwners.contains(key)) {
        const QString &owner = m_keyOwners[key];
        if (m_users.contains(owner)) {
            m_users[owner].licenseStatus = "None";
            m_users[owner].licenseKey.clear();
            m_users[owner].licenseExpiry = QDate();
        }
        m_keyOwners.remove(key);
    }
    m_licenseKeys.remove(key);
    if (m_settings) {
        m_settings->remove("LicenseKeys/" + key + "_used");
        m_settings->remove("LicenseKeys/" + key + "_user");
        saveUsers();
        saveLicenseKeys();
        m_settings->sync();
    }
    return true;
}

// ─── SMTP Config ──────────────────────────────────────────────────────────────

QString UserManager::smtpSenderEmail() const {
    std::shared_lock lock(m_mutex);
    return m_settings ? m_settings->value("SMTP/sender_email").toString() : QString();
}

QString UserManager::smtpSenderPassword() const {
    std::shared_lock lock(m_mutex);
    return m_settings ? m_settings->value("SMTP/sender_password").toString() : QString();
}

void UserManager::setSmtpCredentials(const QString &email, const QString &password) {
    std::unique_lock lock(m_mutex);
    if (!m_settings) return;
    m_settings->setValue("SMTP/sender_email",    email);
    m_settings->setValue("SMTP/sender_password", password);
    m_settings->sync();
}

// ─── Per-user Preferences ─────────────────────────────────────────────────────

QString UserManager::getUserPref(const QString &username, const QString &key, const QString &defaultVal) const {
    std::shared_lock lock(m_mutex);
    if (!m_settings || username.isEmpty()) return defaultVal;
    return m_settings->value("User_" + username + "/pref_" + key, defaultVal).toString();
}

void UserManager::setUserPref(const QString &username, const QString &key, const QString &value) {
    std::unique_lock lock(m_mutex);
    if (!m_settings || username.isEmpty()) return;
    m_settings->setValue("User_" + username + "/pref_" + key, value);
    m_settings->sync();
}
