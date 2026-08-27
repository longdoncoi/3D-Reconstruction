#ifndef MAIL_FILTER_DIALOG_H
#define MAIL_FILTER_DIALOG_H

#include <QStringList>

class IAppContext;
class QWidget;

namespace MailFilterDialog {

bool edit(QWidget *parent, IAppContext *context, QStringList &filterKeywords);

}

#endif // MAIL_FILTER_DIALOG_H
