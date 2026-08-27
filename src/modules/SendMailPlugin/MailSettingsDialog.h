#ifndef MAIL_SETTINGS_DIALOG_H
#define MAIL_SETTINGS_DIALOG_H

class IAppContext;
class IMailService;
class QWidget;

namespace MailSettingsDialog {

void show(QWidget *parent, IAppContext *context, IMailService *mailService);

}

#endif // MAIL_SETTINGS_DIALOG_H
