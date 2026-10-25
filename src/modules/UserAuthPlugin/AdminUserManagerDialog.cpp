#include "AdminUserManagerDialog.h"
#include "IUserManager.h"
#include "UserAuthPlugin.h"
#include "../../utils/ModernMessageBox.h"
#include "StyleManager.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QInputDialog>
#include <QLineEdit>
#include <QComboBox>
#include <QDialog>
#include <QFormLayout>
#include <QClipboard>
#include <QApplication>

AdminUserManagerDialog::AdminUserManagerDialog(IUserManager *users, QWidget *parent)
    : QWidget(parent)
    , m_users(users)
{
    setMinimumSize(800, 600);
    setupUi();
    applyStyle();
    refreshTable();
}

void AdminUserManagerDialog::setupUi() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    auto *tabs = new QTabWidget(this);

    // ── Tab 1: Users ──────────────────────────────────────────────────────────
    auto *usersTab = new QWidget;
    auto *uvl = new QVBoxLayout(usersTab);
    uvl->setContentsMargins(8, 8, 8, 8);
    uvl->setSpacing(8);

    // Action buttons row (top-left, before table)
    auto *userBtnLayout = new QHBoxLayout;
    userBtnLayout->setSpacing(8);
    auto *addBtn    = new QPushButton("➕ " + UserAuthPlugin::translate("admin.add_user"), usersTab);
    m_editBtn       = new QPushButton("✏️ " + UserAuthPlugin::translate("admin.edit_user"), usersTab);
    m_deleteBtn     = new QPushButton("🗑 " + UserAuthPlugin::translate("admin.delete_user"), usersTab);
    for (auto *b : {addBtn, m_editBtn, m_deleteBtn}) {
        b->setFixedHeight(34);
        userBtnLayout->addWidget(b);
    }
    userBtnLayout->addStretch(); // Push buttons to the left
    m_editBtn->setEnabled(false);
    m_deleteBtn->setEnabled(false);
    uvl->addLayout(userBtnLayout);  // Buttons ABOVE table

    m_userTable = new QTableWidget(0, 6, usersTab);
    m_userTable->setHorizontalHeaderLabels({UserAuthPlugin::translate("admin.col_username"), UserAuthPlugin::translate("admin.col_email"), UserAuthPlugin::translate("admin.col_role"),
                                             UserAuthPlugin::translate("admin.col_status"), "License Key", UserAuthPlugin::translate("admin.col_expiry")});
    m_userTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_userTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_userTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_userTable->setAlternatingRowColors(true);
    m_userTable->verticalHeader()->setVisible(false);
    uvl->addWidget(m_userTable, 1);  // Table fills remaining space
    tabs->addTab(usersTab, "👥 " + UserAuthPlugin::translate("admin.tab_users"));

    connect(addBtn,        &QPushButton::clicked, this, &AdminUserManagerDialog::onAddUser);
    connect(m_editBtn,     &QPushButton::clicked, this, &AdminUserManagerDialog::onEditUser);
    connect(m_deleteBtn,   &QPushButton::clicked, this, &AdminUserManagerDialog::onDeleteUser);
    connect(m_userTable,   &QTableWidget::itemSelectionChanged,
            this, &AdminUserManagerDialog::onTableSelectionChanged);

    // ── Tab 2: License Keys ───────────────────────────────────────────────────
    auto *keysTab = new QWidget;
    auto *kvl = new QVBoxLayout(keysTab);

    m_keyTable = new QTableWidget(0, 4, keysTab);
    m_keyTable->setHorizontalHeaderLabels({UserAuthPlugin::translate("admin.col_license"), UserAuthPlugin::translate("admin.col_status"), UserAuthPlugin::translate("admin.tab_users"), ""});
    m_keyTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_keyTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_keyTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_keyTable->setAlternatingRowColors(true);
    m_keyTable->verticalHeader()->setVisible(false);
    kvl->addWidget(m_keyTable);

    auto *keyBtnLayout = new QHBoxLayout;
    auto *genBtn    = new QPushButton("🔑 " + UserAuthPlugin::translate("admin.gen_key"), keysTab);
    m_revokeBtn     = new QPushButton("❌ " + UserAuthPlugin::translate("admin.revoke_key"), keysTab);
    for (auto *b : {genBtn, m_revokeBtn}) {
        b->setFixedHeight(36);
        keyBtnLayout->addWidget(b);
    }
    kvl->addLayout(keyBtnLayout);
    mainLayout->addWidget(tabs);
    tabs->addTab(keysTab, "🔑 License Keys");

    connect(genBtn,     &QPushButton::clicked, this, &AdminUserManagerDialog::onGenerateLicenseKey);
    connect(m_revokeBtn,&QPushButton::clicked, this, &AdminUserManagerDialog::onRevokeLicenseKey);
    connect(m_keyTable, &QTableWidget::cellClicked, this, &AdminUserManagerDialog::onKeyTableCellClicked);

    // ── Tab 3: SMTP Settings ──────────────────────────────────────────────────
    auto *smtpTab = new QWidget;
    auto *svl = new QVBoxLayout(smtpTab);
    svl->setContentsMargins(20, 20, 20, 20);
    svl->setSpacing(12);
    svl->addWidget(new QLabel(UserAuthPlugin::translate("smtp.desc"), smtpTab));
    svl->addWidget(new QLabel(UserAuthPlugin::translate("smtp.gmail_hint"), smtpTab));
    svl->addWidget(new QLabel(UserAuthPlugin::translate("smtp.outlook_hint"), smtpTab));

    auto *smtpForm = new QFormLayout;
    auto *smtpEmailEdit = new QLineEdit(m_users->smtpSenderEmail(), smtpTab);
    smtpEmailEdit->setObjectName("inputField");
    smtpEmailEdit->setFixedHeight(38);
    auto *smtpPassEdit = new QLineEdit(smtpTab);
    smtpPassEdit->setObjectName("inputField");
    smtpPassEdit->setEchoMode(QLineEdit::Password);
    smtpPassEdit->setPlaceholderText(UserAuthPlugin::translate("smtp.pass_placeholder"));
    smtpPassEdit->setFixedHeight(38);
    smtpForm->addRow(UserAuthPlugin::translate("smtp.lbl_email"), smtpEmailEdit);
    smtpForm->addRow(UserAuthPlugin::translate("smtp.lbl_password"), smtpPassEdit);
    svl->addLayout(smtpForm);

    auto *saveSmtpBtn = new QPushButton("💾 " + UserAuthPlugin::translate("smtp.save_btn"), smtpTab);
    saveSmtpBtn->setFixedHeight(38);
    svl->addWidget(saveSmtpBtn, 0, Qt::AlignLeft);
    svl->addStretch();
    tabs->addTab(smtpTab, "📧 " + UserAuthPlugin::translate("smtp.tab_title"));

    connect(saveSmtpBtn, &QPushButton::clicked, this, [=]() {
        QString pass = smtpPassEdit->text().trimmed();
        pass.remove(' '); // Loại bỏ dấu cách trong App Password nếu copy nhầm
        // Empty field = keep the stored password; re-type is required to change
        // it, and the secret is never echoed back into the UI.
        if (pass.isEmpty())
            pass = m_users->smtpSenderPassword();
        m_users->setSmtpCredentials(smtpEmailEdit->text().trimmed(), pass);
        smtpPassEdit->clear();
        ModernMessageBox::information(this, UserAuthPlugin::translate("common.success"), UserAuthPlugin::translate("smtp.save_success"));
    });

    mainLayout->addWidget(tabs);
}

void AdminUserManagerDialog::applyStyle() {
    // Relying on global stylesheet from StyleManager for a unified look.
    // Specific tweaks only if necessary (using object names).
}

void AdminUserManagerDialog::refreshTable() {
    auto *um = m_users;

    // Refresh user table
    QStringList names = um->allUsernames();
    m_userTable->setRowCount(names.size());
    for (int i = 0; i < names.size(); ++i) {
        UserInfo u = um->userInfo(names[i]);
        m_userTable->setItem(i, 0, new QTableWidgetItem(u.username));
        m_userTable->setItem(i, 1, new QTableWidgetItem(u.email));
        m_userTable->setItem(i, 2, new QTableWidgetItem(u.role == UserRole::Admin ? "🛡 Admin" : UserAuthPlugin::translate("admin.role_user")));
        m_userTable->setItem(i, 3, new QTableWidgetItem(u.licenseStatus));
        m_userTable->setItem(i, 4, new QTableWidgetItem(u.licenseKey));
        m_userTable->setItem(i, 5, new QTableWidgetItem(
            u.licenseExpiry.isValid() ? u.licenseExpiry.toString("dd/MM/yyyy") : "—"));
    }

    // Refresh key table
    QStringList keys = um->availableLicenseKeys();
    m_keyTable->setRowCount(keys.size());
    for (int i = 0; i < keys.size(); ++i) {
        // Find owner via checking users
        QString owner, status = UserAuthPlugin::translate("admin.status_unused");
        for (const QString &n : names) {
            UserInfo u = um->userInfo(n);
            if (u.licenseKey == keys[i]) { owner = n; status = UserAuthPlugin::translate("admin.status_used"); break; }
        }
        m_keyTable->setItem(i, 0, new QTableWidgetItem(keys[i]));
        m_keyTable->setItem(i, 1, new QTableWidgetItem(status));
        m_keyTable->setItem(i, 2, new QTableWidgetItem(owner));
        
        // Copy button styled as a premium clickable link
        auto *copyItem = new QTableWidgetItem("📋 " + UserAuthPlugin::translate("common.copy"));
        copyItem->setForeground(QColor(StyleManager::accentColor()));
        copyItem->setTextAlignment(Qt::AlignCenter);
        m_keyTable->setItem(i, 3, copyItem);
    }
}

void AdminUserManagerDialog::onTableSelectionChanged() {
    bool sel = !m_userTable->selectedItems().isEmpty();
    m_editBtn->setEnabled(sel);
    m_deleteBtn->setEnabled(sel);
}

void AdminUserManagerDialog::onAddUser() {
    ModernDialog dlg(UserAuthPlugin::translate("admin.add_user_title"), this);
    dlg.setFixedSize(400, 320);
    
    auto *content = new QWidget(&dlg);
    auto *fl = new QFormLayout(content);
    fl->setContentsMargins(20,20,20,20);
    fl->setSpacing(15);

    auto *usernameEdit = new QLineEdit(content);
    auto *emailEdit    = new QLineEdit(content);
    auto *passwordEdit = new QLineEdit(content); 
    passwordEdit->setEchoMode(QLineEdit::Password);
    auto *roleCombo    = new QComboBox(content);
    roleCombo->addItems({"User", "Admin"});

    fl->addRow(UserAuthPlugin::translate("admin.lbl_username"), usernameEdit);
    fl->addRow(UserAuthPlugin::translate("common.email"), emailEdit);
    fl->addRow(UserAuthPlugin::translate("admin.lbl_password"), passwordEdit);
    fl->addRow(UserAuthPlugin::translate("admin.lbl_role"), roleCombo);

    auto *hl = new QHBoxLayout;
    auto *okBtn  = new QPushButton(UserAuthPlugin::translate("admin.add_now"), content);
    okBtn->setObjectName("primary");
    auto *canBtn = new QPushButton(UserAuthPlugin::translate("common.cancel"), content);
    hl->addStretch();
    hl->addWidget(canBtn); hl->addWidget(okBtn);
    fl->addRow(hl);

    connect(okBtn,  &QPushButton::clicked, &dlg, &QDialog::accept);
    connect(canBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    
    dlg.setContentLayout(fl);

    if (dlg.exec() != QDialog::Accepted) return;

    UserInfo u;
    u.username = usernameEdit->text().trimmed();
    u.email    = emailEdit->text().trimmed();
    u.role     = roleCombo->currentText() == "Admin" ? UserRole::Admin : UserRole::User;
    QString error;
    if (!m_users->addUser(u, passwordEdit->text(), error))
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), error);
    refreshTable();
}

void AdminUserManagerDialog::onEditUser() {
    int row = m_userTable->currentRow();
    if (row < 0) return;
    QString username = m_userTable->item(row, 0)->text();
    UserInfo u = m_users->userInfo(username);

    ModernDialog dlg(UserAuthPlugin::translate("admin.edit_user_title") + username, this);
    dlg.setFixedSize(400, 260);
    
    auto *content = new QWidget(&dlg);
    auto *fl = new QFormLayout(content);
    fl->setContentsMargins(20,20,20,20);
    fl->setSpacing(15);

    auto *emailEdit = new QLineEdit(u.email, content);
    auto *roleCombo = new QComboBox(content);
    roleCombo->addItems({"User", "Admin"});
    roleCombo->setCurrentText(u.role == UserRole::Admin ? "Admin" : "User");

    fl->addRow(UserAuthPlugin::translate("common.email"), emailEdit);
    fl->addRow(UserAuthPlugin::translate("admin.lbl_role"), roleCombo);

    auto *hl = new QHBoxLayout;
    auto *okBtn  = new QPushButton(UserAuthPlugin::translate("admin.save_changes"), content);
    okBtn->setObjectName("primary");
    auto *canBtn = new QPushButton(UserAuthPlugin::translate("common.cancel"), content);
    hl->addStretch();
    hl->addWidget(canBtn); hl->addWidget(okBtn);
    fl->addRow(hl);

    connect(okBtn,  &QPushButton::clicked, &dlg, &QDialog::accept);
    connect(canBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    
    dlg.setContentLayout(fl);

    if (dlg.exec() != QDialog::Accepted) return;

    u.email = emailEdit->text().trimmed();
    u.role  = roleCombo->currentText() == "Admin" ? UserRole::Admin : UserRole::User;
    QString err;
    if (!m_users->updateUser(u, err))
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), err);
    refreshTable();
}

void AdminUserManagerDialog::onDeleteUser() {
    int row = m_userTable->currentRow();
    if (row < 0) return;
    QString username = m_userTable->item(row, 0)->text();
    if (ModernMessageBox::question(UserAuthPlugin::languageService(), this, UserAuthPlugin::translate("common.confirm"), UserAuthPlugin::translate("admin.del_confirm_msg") + username + "'?")) {
        QString error;
        if (!m_users->deleteUser(username, error))
            ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), error);
        refreshTable();
    }
}

void AdminUserManagerDialog::onGenerateLicenseKey() {
    AuthError code = AuthError::None;
    QString key = m_users->generateLicenseKey(&code);
    if (key.isEmpty()) {
        // The core refuses key generation without the administrator role.
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"),
                                  UserAuthPlugin::translate("auth.not_authorized"));
        return;
    }
    ModernMessageBox::information(this,
        UserAuthPlugin::translate("admin.new_key_title"),
        UserAuthPlugin::translate("admin.new_key_msg").arg(key));
    QApplication::clipboard()->setText(key);
    refreshTable();
}

void AdminUserManagerDialog::onRevokeLicenseKey() {
    int row = m_keyTable->currentRow();
    if (row < 0) {
        ModernMessageBox::warning(this, UserAuthPlugin::translate("admin.not_selected"), UserAuthPlugin::translate("admin.select_key_revoke"));
        return;
    }
    QString key = m_keyTable->item(row, 0)->text();
    if (!ModernMessageBox::question(UserAuthPlugin::languageService(), this, UserAuthPlugin::translate("common.confirm"),
        UserAuthPlugin::translate("admin.revoke_key_msg").arg(key))) return;
    QString error;
    if (!m_users->revokeLicenseKey(key, error))
        ModernMessageBox::warning(this, UserAuthPlugin::translate("common.error"), error);
    refreshTable();
}

void AdminUserManagerDialog::onSmtpSettings() {
    // Handled inline in SMTP tab
}

void AdminUserManagerDialog::onKeyTableCellClicked(int row, int col) {
    if (col == 3) {
        QTableWidgetItem *item = m_keyTable->item(row, 0);
        if (item) {
            QString key = item->text().trimmed();
            if (!key.isEmpty()) {
                QApplication::clipboard()->setText(key);
                ModernMessageBox::information(this, UserAuthPlugin::translate("common.copy"),
                    UserAuthPlugin::translate("admin.copied_key_msg").arg(key));
            }
        }
    }
}
