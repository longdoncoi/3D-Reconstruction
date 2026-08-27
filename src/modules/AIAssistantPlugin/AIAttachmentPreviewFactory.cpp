#include "AIAttachmentPreviewFactory.h"

#include <QLabel>
#include <QPushButton>
#include <QWidget>

namespace AIAttachmentPreviewFactory {

QWidget *create(QWidget *parent,
                const FileUtilities::AttachmentResult &attachment,
                const std::function<void(QWidget *)> &removeCallback)
{
    auto *previewWidget = new QWidget(parent);
    previewWidget->setFixedSize(84, 84);
    previewWidget->setStyleSheet("background:#2a2a35; border-radius:6px; border:1px solid #3a3a4a;");

    auto *imageLabel = new QLabel(previewWidget);
    imageLabel->setGeometry(2, 2, 80, 80);
    imageLabel->setPixmap(attachment.thumbnail);
    imageLabel->setAlignment(Qt::AlignCenter);

    auto *removeButton = new QPushButton("×", previewWidget);
    removeButton->setGeometry(64, 2, 18, 18);
    removeButton->setStyleSheet(
        "QPushButton { background:rgba(0,0,0,150); color:white; border-radius:9px; "
        "font-weight:bold; font-size:12px; padding-bottom:2px; }"
        "QPushButton:hover { background:rgba(255,50,50,200); }");

    QObject::connect(removeButton, &QPushButton::clicked, previewWidget, [removeCallback, previewWidget]() {
        removeCallback(previewWidget);
    });

    return previewWidget;
}

}
