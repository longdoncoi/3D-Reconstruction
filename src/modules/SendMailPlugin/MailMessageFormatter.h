#ifndef MAIL_MESSAGE_FORMATTER_H
#define MAIL_MESSAGE_FORMATTER_H

#include "IMailService.h"

#include <QString>

namespace MailMessageFormatter {

QString senderName(const QString &from);
QString previewHtml(const MailMessage &message, const QString &emptyBodyText);

}

#endif // MAIL_MESSAGE_FORMATTER_H
