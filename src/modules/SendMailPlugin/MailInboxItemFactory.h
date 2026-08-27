#ifndef MAIL_INBOX_ITEM_FACTORY_H
#define MAIL_INBOX_ITEM_FACTORY_H

#include "IMailService.h"

class QListWidget;

namespace MailInboxItemFactory {

void addMessageItem(QListWidget *list, const MailMessage &message, int row, const QString &fallbackSubject);

}

#endif // MAIL_INBOX_ITEM_FACTORY_H
