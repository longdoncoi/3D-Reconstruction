#include "AvatarCropperDialog.h"
#include "AppConfig.h"
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QVBoxLayout>
#include <QPushButton>
#include <QHBoxLayout>
#include <QListWidget>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QStandardPaths>
#include <QLabel>
#include <QMenu>
#include <QEvent>
#include "UserManager.h"
#include "LanguageManager.h"
#include "UserAuthPlugin.h"

AvatarCropperDialog::AvatarCropperDialog(const QString& username, QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("avatar.cropper_title"), parent), m_username(username)
{
    setFixedSize(700, 500);

    auto *content = new QWidget(this);
    QHBoxLayout *mainLayout = new QHBoxLayout(content);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    mainLayout->setSpacing(15);
    
    // --- Left: Canvas widget for drawing ---
    m_canvas = new QWidget(content);
    m_canvas->setFixedSize(400, 400);
    m_canvas->setObjectName("avatarCanvas");
    m_canvas->setStyleSheet("#avatarCanvas { background-color: #1a1a2e; border: 2px solid #ef4444; border-radius: 8px; }");
    m_canvas->installEventFilter(this);

    QVBoxLayout *leftLayout = new QVBoxLayout();
    leftLayout->setAlignment(Qt::AlignTop | Qt::AlignHCenter);
    leftLayout->addWidget(m_canvas, 0, Qt::AlignHCenter);
    
    QHBoxLayout *btnLayout = new QHBoxLayout();
    QPushButton* cancelBtn = new QPushButton(UserAuthPlugin::translate("common.cancel"), content);
    QPushButton* saveBtn = new QPushButton(UserAuthPlugin::translate("common.save"), content);
    saveBtn->setObjectName("primary");
    
    btnLayout->addStretch();
    btnLayout->addWidget(cancelBtn);
    btnLayout->addWidget(saveBtn);
    leftLayout->addLayout(btnLayout);
    
    QVBoxLayout *rightLayout = new QVBoxLayout();
    QLabel *historyLabel = new QLabel(UserAuthPlugin::translate("avatar.history"), content);
    historyLabel->setStyleSheet("color: #94a3b8; font-weight: bold;");
    rightLayout->addWidget(historyLabel);

    m_historyList = new QListWidget(content);
    m_historyList->setObjectName("avatarHistoryList");
    m_historyList->setContextMenuPolicy(Qt::CustomContextMenu);
    m_historyList->setIconSize(QSize(48, 48));
    m_historyList->setStyleSheet(
        "QListWidget::item { padding: 4px; }"
        "QListWidget::item:selected { background-color: #3b82f6; color: white; border-radius: 4px; }"
        "QListWidget::item:selected:!active { background-color: #3b82f6; color: white; border-radius: 4px; }"
        "QListWidget::item:hover { background-color: #2c2c3e; border-radius: 4px; }"
    );
    rightLayout->addWidget(m_historyList);

    m_uploadBtn = new QPushButton("📤 " + UserAuthPlugin::translate("avatar.upload_btn"), content);
    m_uploadBtn->setCursor(Qt::PointingHandCursor);
    m_uploadBtn->setFixedHeight(40);
    rightLayout->addWidget(m_uploadBtn);

    mainLayout->addLayout(leftLayout, 1);
    mainLayout->addLayout(rightLayout, 0);

    connect(saveBtn, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    connect(m_uploadBtn, &QPushButton::clicked, this, &AvatarCropperDialog::onUpload);
    connect(m_historyList, &QListWidget::itemClicked, this, &AvatarCropperDialog::onAvatarSelected);
    connect(m_historyList, &QListWidget::customContextMenuRequested, this, [this](const QPoint &pos) {
        QListWidgetItem *item = m_historyList->itemAt(pos);
        if (!item) return;
        QMenu menu(this);
        QAction *delAction = menu.addAction(UserAuthPlugin::translate("avatar.delete_img"));
        connect(delAction, &QAction::triggered, this, &AvatarCropperDialog::onDeleteAvatar);
        menu.exec(m_historyList->mapToGlobal(pos));
    });

    setContentLayout(mainLayout);
    loadHistory();
}

void AvatarCropperDialog::accept() {
    if (m_sourceImage.isNull()) {
        QDialog::reject();
        return;
    }

    // 1. Save original to history if not already there
    QString uploadDir = AppConfig::instance().uploadDir() + "/Avatars/" + m_username;
    QDir().mkpath(uploadDir);
    
    QFileInfo fi(m_currentImagePath);
    QString destPath = uploadDir + "/" + fi.fileName();
    
    if (QDir::toNativeSeparators(m_currentImagePath) != QDir::toNativeSeparators(destPath)) {
        if (QFile::exists(destPath)) {
            // Add timestamp to avoid overwrite if same name but different file
            destPath = uploadDir + "/" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_") + fi.fileName();
        }
        QFile::copy(m_currentImagePath, destPath);
    }

    QString metaFile = uploadDir + "/_current_avatar.txt";
    QFile meta(metaFile);
    if (meta.open(QIODevice::WriteOnly)) {
        meta.write(destPath.toUtf8());
        meta.close();
    }

    // 2. Save Cropped Thumbnail
    QPixmap cropped = getCroppedAvatar();
    
    QString thumbDir = AppConfig::instance().thumbnailsDir() + "/Avatars";
    QDir().mkpath(thumbDir);
    QString thumbFile = thumbDir + "/" + m_username + ".png";

    if (QFile::exists(thumbFile)) {
        QFile::remove(thumbFile);
    }

    if (cropped.save(thumbFile)) {
        UserManager::instance()->updateAvatar(m_username, thumbFile);
    }

    QDialog::accept();
}

void AvatarCropperDialog::loadHistory() {
    m_historyList->clear();
    QString uploadDir = AppConfig::instance().uploadDir() + "/Avatars/" + m_username;
    QDir dir(uploadDir);
    
    if (dir.exists()) {
        QStringList files = dir.entryList({"*.png", "*.jpg", "*.jpeg"}, QDir::Files, QDir::Time);
        for (const QString &f : files) {
            QPixmap pix(dir.absoluteFilePath(f));
            if (!pix.isNull()) {
                int sz = qMin(pix.width(), pix.height());
                QRect cropRect((pix.width() - sz) / 2, (pix.height() - sz) / 2, sz, sz);
                QIcon icon(pix.copy(cropRect).scaled(48, 48, Qt::KeepAspectRatio, Qt::SmoothTransformation));
                QListWidgetItem *item = new QListWidgetItem(icon, f, m_historyList);
                item->setData(Qt::UserRole, dir.absoluteFilePath(f));
            }
        }
    }

    QString currentAvatar = UserManager::instance()->userInfo(m_username).avatarPath;
    
    QString metaFile = uploadDir + "/_current_avatar.txt";
    QString originalAvatarPath;
    QFile meta(metaFile);
    if (meta.open(QIODevice::ReadOnly)) {
        originalAvatarPath = QString::fromUtf8(meta.readAll());
        meta.close();
    }

    bool selected = false;
    if (!originalAvatarPath.isEmpty() && QFile::exists(originalAvatarPath)) {
        for (int i = 0; i < m_historyList->count(); ++i) {
            if (m_historyList->item(i)->data(Qt::UserRole).toString() == originalAvatarPath) {
                m_historyList->setCurrentRow(i);
                onAvatarSelected();
                selected = true;
                break;
            }
        }
    }
    
    if (!selected) {
        if (m_historyList->count() > 0) {
            // Prioritize the history list to load the original, un-rounded image
            m_historyList->setCurrentRow(0);
            onAvatarSelected();
        } else if (!currentAvatar.isEmpty() && QFile::exists(currentAvatar)) {
            // Fallback to the current avatar (which is the cropped thumbnail)
            m_currentImagePath = currentAvatar;
            loadImage(currentAvatar);
        } else {
            // Create default avatar placeholder
        QImage def(512, 512, QImage::Format_ARGB32);
        def.fill(Qt::transparent);
        QPainter p(&def);
        p.setRenderHint(QPainter::Antialiasing);
        p.setBrush(QColor("#7c3aed"));
        p.setPen(Qt::NoPen);
        p.drawEllipse(0, 0, 512, 512);
        p.setPen(Qt::white);
        p.setFont(QFont("Segoe UI", 200, QFont::Bold));
        p.drawText(def.rect(), Qt::AlignCenter, m_username.isEmpty() ? "?" : m_username.left(1).toUpper());
        
        m_sourceImage = def;
        m_currentImagePath.clear();
        loadImage(QString()); // Call with empty string to trigger view reset but not reload
    }
}
}

void AvatarCropperDialog::onUpload() {
    QString fileName = QFileDialog::getOpenFileName(this, UserAuthPlugin::translate("file.select_image"),
        QStandardPaths::writableLocation(QStandardPaths::PicturesLocation),
        "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty()) {
        m_currentImagePath = fileName;
        loadImage(fileName);
    }
}

void AvatarCropperDialog::onAvatarSelected() {
    QListWidgetItem *item = m_historyList->currentItem();
    if (item) {
        QString path = item->data(Qt::UserRole).toString();
        m_currentImagePath = path;
        loadImage(path);
    }
}

void AvatarCropperDialog::onDeleteAvatar() {
    QListWidgetItem *item = m_historyList->currentItem();
    if (!item) return;

    QString path = item->data(Qt::UserRole).toString();
    if (QFile::remove(path)) {
        if (m_currentImagePath == path) {
            m_currentImagePath.clear();
        }
        loadHistory();
        update();
    }
}

void AvatarCropperDialog::loadImage(const QString &path) {
    if (!path.isEmpty()) {
        m_sourceImage.load(path);
    }

    if (!m_sourceImage.isNull()) {
        double scaleX = 400.0 / m_sourceImage.width();
        double scaleY = 400.0 / m_sourceImage.height();
        m_scale = qMax(scaleX, scaleY);

        m_offset.setX((400 - m_sourceImage.width() * m_scale) / 2.0);
        m_offset.setY((400 - m_sourceImage.height() * m_scale) / 2.0);
    }
    if (m_canvas) m_canvas->update();
    update();
}

QPixmap AvatarCropperDialog::getCroppedAvatar() const {
    QImage result(m_cropSize, m_cropSize, QImage::Format_ARGB32);
    result.fill(Qt::transparent);
    QPainter p(&result);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    
    QPainterPath path;
    path.addEllipse(0, 0, m_cropSize, m_cropSize);
    p.setClipPath(path);

    double cx = 200.0;
    double cy = 200.0;
    double dx = cx - m_cropSize / 2.0;
    double dy = cy - m_cropSize / 2.0;

    p.translate(-dx, -dy);
    p.translate(m_offset.x(), m_offset.y());
    p.scale(m_scale, m_scale);
    p.drawImage(0, 0, m_sourceImage);
    
    return QPixmap::fromImage(result);
}

void AvatarCropperDialog::paintEvent(QPaintEvent *event) {
    // Do nothing at dialog level - painting is done in eventFilter on m_canvas
    QDialog::paintEvent(event);
}

bool AvatarCropperDialog::eventFilter(QObject *obj, QEvent *event) {
    if (obj != m_canvas) return QDialog::eventFilter(obj, event);

    if (event->type() == QEvent::Paint) {
        QPainter p(m_canvas);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        p.fillRect(m_canvas->rect(), QColor("#1a1a2e"));

        if (m_sourceImage.isNull()) {
            p.setPen(QColor("#888"));
            p.drawText(m_canvas->rect(), Qt::AlignCenter, UserAuthPlugin::translate("avatar.guide"));
        } else {
            p.save();
            p.setClipRect(m_canvas->rect());
            p.translate(m_offset.x(), m_offset.y());
            p.scale(m_scale, m_scale);
            p.drawImage(0, 0, m_sourceImage);
            p.restore();

            // Crop guide overlay
            QPainterPath bg;
            bg.addRect(m_canvas->rect());
            QPainterPath hole;
            hole.addEllipse(200 - m_cropSize / 2.0, 200 - m_cropSize / 2.0, m_cropSize, m_cropSize);
            bg = bg.subtracted(hole);
            p.fillPath(bg, QColor(0, 0, 0, 160));
            p.setPen(QPen(QColor("#7c3aed"), 2));
            p.drawEllipse(200 - m_cropSize / 2.0, 200 - m_cropSize / 2.0, m_cropSize, m_cropSize);
        }
        return true;
    }

    if (event->type() == QEvent::MouseButtonPress) {
        auto *me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::LeftButton) {
            m_isDragging = true;
            m_lastMousePos = me->pos();
        }
        return true;
    }
    if (event->type() == QEvent::MouseMove) {
        auto *me = static_cast<QMouseEvent*>(event);
        if (m_isDragging) {
            QPoint delta = me->pos() - m_lastMousePos;
            m_offset += delta;
            m_lastMousePos = me->pos();
            m_canvas->update();
        }
        return true;
    }
    if (event->type() == QEvent::MouseButtonRelease) {
        m_isDragging = false;
        return true;
    }
    if (event->type() == QEvent::Wheel) {
        auto *we = static_cast<QWheelEvent*>(event);
        double scaleFactor = (we->angleDelta().y() > 0) ? 1.1 : (1.0 / 1.1);
        QPointF mousePos = we->position();
        m_offset = mousePos - (mousePos - m_offset) * scaleFactor;
        m_scale *= scaleFactor;
        m_canvas->update();
        return true;
    }

    return QDialog::eventFilter(obj, event);
}

void AvatarCropperDialog::mousePressEvent(QMouseEvent *event) {
    // Mouse events on canvas are handled by eventFilter
    ModernDialog::mousePressEvent(event);
}

void AvatarCropperDialog::mouseMoveEvent(QMouseEvent *event) {
    ModernDialog::mouseMoveEvent(event);
}

void AvatarCropperDialog::mouseReleaseEvent(QMouseEvent *event) {
    m_isDragging = false;
    ModernDialog::mouseReleaseEvent(event);
}

void AvatarCropperDialog::wheelEvent(QWheelEvent *event) {
    // Handled in eventFilter
    event->ignore();
}
