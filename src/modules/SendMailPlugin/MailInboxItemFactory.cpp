#include "MailInboxItemFactory.h"

#include "MailMessageFormatter.h"

#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QVBoxLayout>

namespace MailInboxItemFactory {

void addMessageItem(QListWidget *list, const MailMessage &message, int row, const QString &fallbackSubject)
{
    auto *itemWidget = new QWidget(list);
    auto *itemLayout = new QVBoxLayout(itemWidget);
    itemLayout->setContentsMargins(12, 8, 12, 8);
    itemLayout->setSpacing(3);

    const QString subjectText = message.subject.isEmpty() ? fallbackSubject : message.subject;
    const QString unreadDot = message.isRead ? QString() : QString::fromUtf8("\xE2\x97\x8F ");
    auto *subjectLabel = new QLabel(unreadDot + subjectText, itemWidget);
    subjectLabel->setStyleSheet(
        message.isRead
            ? "color: #e2e8f0; font-size: 13px;"
            : "color: #f1f5f9; font-size: 13px; font-weight: bold;");
    subjectLabel->setWordWrap(false);
    subjectLabel->setTextFormat(Qt::PlainText);
    itemLayout->addWidget(subjectLabel);

    auto *senderLabel = new QLabel(MailMessageFormatter::senderName(message.from), itemWidget);
    senderLabel->setStyleSheet("color: #94a3b8; font-size: 11px;");
    senderLabel->setTextFormat(Qt::PlainText);
    itemLayout->addWidget(senderLabel);

    const QString dateText = message.date.isValid()
        ? message.date.toString("dd/MM/yyyy  hh:mm")
        : QString();
    if (!dateText.isEmpty()) {
        auto *dateLabel = new QLabel(dateText, itemWidget);
        dateLabel->setStyleSheet("color: #64748b; font-size: 10px;");
        itemLayout->addWidget(dateLabel);
    }

    itemWidget->setLayout(itemLayout);

    auto *item = new QListWidgetItem();
    item->setData(Qt::UserRole, message.uid);
    item->setData(Qt::UserRole + 1, row);
    item->setToolTip(message.from);
    item->setSizeHint(itemWidget->sizeHint());
    list->addItem(item);
    list->setItemWidget(item, itemWidget);
}

}
