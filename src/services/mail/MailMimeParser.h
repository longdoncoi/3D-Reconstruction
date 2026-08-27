#ifndef MAIL_MIME_PARSER_H
#define MAIL_MIME_PARSER_H

#include "IMailService.h"

#include <QString>
#include <QStringList>

namespace MailMimeParser {

QStringList splitAddresses(const QString &value);
QString htmlWithSignature(QString html, const QString &signature);
MailMessage parseFetchedMessage(const QString &uid, const QString &raw);

}

#endif // MAIL_MIME_PARSER_H
