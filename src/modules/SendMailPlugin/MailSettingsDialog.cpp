#include "MailSettingsDialog.h"

#include "IAppContext.h"
#include "IMailService.h"
#include "ModernMessageBox.h"
#include "UserManager.h"

#include <QDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>

namespace MailSettingsDialog {

void show(QWidget *parent, IAppContext *context, IMailService *mailService)
{
    QDialog dlg(parent);
    dlg.setWindowTitle(context->translate("mail.settings"));
    dlg.setMinimumWidth(460);
    auto *layout = new QVBoxLayout(&dlg);
    auto *form = new QFormLayout();

    auto *email = new QLineEdit(mailService ? mailService->senderEmail() : QString(), &dlg);
    auto *password = new QLineEdit(&dlg);
    password->setEchoMode(QLineEdit::Password);
    auto *displayName = new QLineEdit(UserManager::instance()->currentUsername(), &dlg);
    auto *signature = new QTextEdit(&dlg);
    signature->setAcceptRichText(true);
    signature->setMinimumHeight(110);

    const QString username = UserManager::instance()->currentUsername();
    password->setText(UserManager::instance()->getUserPref(username, "mail_password"));
    displayName->setText(UserManager::instance()->getUserPref(username, "mail_display_name", username));
    signature->setHtml(UserManager::instance()->getUserPref(
        username,
        "mail_signature",
        QString("<b>%1</b><br>3D-Reconstruction").arg(username)));

    form->addRow(context->translate("mail.account"), email);
    form->addRow(context->translate("mail.password"), password);
    form->addRow(context->translate("mail.display_name"), displayName);
    form->addRow(context->translate("mail.signature"), signature);
    layout->addLayout(form);

    auto *buttons = new QHBoxLayout();
    auto *testBtn = new QPushButton(context->translate("mail.test"), &dlg);
    auto *cancelBtn = new QPushButton(context->translate("common.cancel"), &dlg);
    auto *saveBtn = new QPushButton(context->translate("common.save"), &dlg);
    saveBtn->setObjectName("primary");
    buttons->addWidget(testBtn);
    buttons->addStretch();
    buttons->addWidget(cancelBtn);
    buttons->addWidget(saveBtn);
    layout->addLayout(buttons);

    QObject::connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    QObject::connect(saveBtn, &QPushButton::clicked, &dlg, [&]() {
        if (mailService) {
            mailService->setCredentials(email->text().trimmed(), password->text(), displayName->text().trimmed());
            UserManager::instance()->setUserPref(username, "mail_signature", signature->toHtml());
        }
        dlg.accept();
    });
    QObject::connect(testBtn, &QPushButton::clicked, &dlg, [&]() {
        if (!mailService) return;
        mailService->setCredentials(email->text().trimmed(), password->text(), displayName->text().trimmed());
        QString error;
        if (mailService->testConnection(error)) {
            ModernMessageBox::information(&dlg, context->translate("common.success"), context->translate("mail.test_ok"));
        } else {
            ModernMessageBox::warning(&dlg, context->translate("common.error"), error);
        }
    });

    dlg.exec();
}

}
