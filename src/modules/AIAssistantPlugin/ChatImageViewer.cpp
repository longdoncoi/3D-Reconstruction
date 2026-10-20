#include "ChatImageViewer.h"

#include "FileUtilities.h"

ChatImageViewer::ChatImageViewer(const QString &imagePath, QWidget *parent) : QDialog(parent) {
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
    setAttribute(Qt::WA_TranslucentBackground);
    setStyleSheet("background-color: rgba(0, 0, 0, 200);");

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    QWidget *topBar = new QWidget(this);
    topBar->setFixedHeight(50);
    QHBoxLayout *topLayout = new QHBoxLayout(topBar);
    topLayout->addStretch();
    QPushButton *btnClose = new QPushButton("×", topBar);
    btnClose->setFixedSize(40, 40);
    btnClose->setStyleSheet(
        "QPushButton { background: transparent; color: white; border: none; font-size: 30px; font-weight: bold; }"
        "QPushButton:hover { color: #ff4d4d; }");
    connect(btnClose, &QPushButton::clicked, this, &QDialog::accept);
    topLayout->addWidget(btnClose);
    layout->addWidget(topBar);

    QLabel *imgLabel = new QLabel(this);
    // The attachment is sealed at rest: it must be unsealed before the pixmap
    // is built — the file on disk is a ciphertext, not an image.
    const QByteArray bytes =
        FileUtilities::readAttachment(imagePath, QString::fromLatin1(FileUtilities::AttachmentContext));
    QPixmap pix;
    if (!bytes.isEmpty()) pix.loadFromData(bytes);
    if (!pix.isNull()) {
        QSize screenSize = screen()->size();
        QSize maxSize(screenSize.width() * 0.8, screenSize.height() * 0.8);
        imgLabel->setPixmap(pix.scaled(maxSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    imgLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(imgLabel);
    
    layout->addStretch();
    
    this->installEventFilter(this);
}

bool ChatImageViewer::eventFilter(QObject *obj, QEvent *event) {
    if (event->type() == QEvent::MouseButtonPress) {
        accept();
        return true;
    }
    return QDialog::eventFilter(obj, event);
}
