#include "MailFilterDialog.h"

#include "IAppContext.h"

#include <QDialog>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>

namespace MailFilterDialog {

bool edit(QWidget *parent, IAppContext *context, QStringList &filterKeywords)
{
    QDialog dlg(parent);
    dlg.setWindowTitle(context->translate("mail.filter"));
    dlg.setMinimumWidth(500);
    dlg.setMinimumHeight(400);
    auto *layout = new QVBoxLayout(&dlg);

    auto *instruction = new QLabel(context->translate("mail.filter_hint"), &dlg);
    instruction->setStyleSheet("color: #888; font-size: 11px;");
    layout->addWidget(instruction);

    auto *listLabel = new QLabel(context->translate("mail.filter_list"), &dlg);
    auto *filterList = new QListWidget(&dlg);
    for (const QString &keyword : filterKeywords) {
        filterList->addItem(keyword);
    }
    layout->addWidget(listLabel);
    layout->addWidget(filterList, 1);

    auto *btnRow = new QHBoxLayout();
    auto *addBtn = new QPushButton(context->translate("mail.add_filter"), &dlg);
    auto *removeBtn = new QPushButton(context->translate("mail.remove_filter"), &dlg);
    btnRow->addWidget(addBtn);
    btnRow->addWidget(removeBtn);
    btnRow->addStretch();
    layout->addLayout(btnRow);

    QObject::connect(addBtn, &QPushButton::clicked, &dlg, [&]() {
        bool ok = false;
        QString keyword = QInputDialog::getText(&dlg,
                                                context->translate("mail.add_filter"),
                                                context->translate("mail.filter_input_hint"),
                                                QLineEdit::Normal,
                                                "",
                                                &ok);
        if (ok && !keyword.trimmed().isEmpty()) {
            keyword = keyword.trimmed();
            if (!filterKeywords.contains(keyword, Qt::CaseInsensitive)) {
                filterKeywords << keyword;
                filterList->addItem(keyword);
            }
        }
    });

    QObject::connect(removeBtn, &QPushButton::clicked, &dlg, [&]() {
        const int row = filterList->currentRow();
        if (row >= 0 && row < filterKeywords.size()) {
            filterKeywords.removeAt(row);
            delete filterList->takeItem(row);
        }
    });

    auto *buttons = new QHBoxLayout();
    auto *cancelBtn = new QPushButton(context->translate("common.cancel"), &dlg);
    auto *applyBtn = new QPushButton(context->translate("common.save"), &dlg);
    applyBtn->setObjectName("primary");
    buttons->addStretch();
    buttons->addWidget(cancelBtn);
    buttons->addWidget(applyBtn);
    layout->addLayout(buttons);

    QObject::connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    QObject::connect(applyBtn, &QPushButton::clicked, &dlg, &QDialog::accept);

    return dlg.exec() == QDialog::Accepted;
}

}
