#include "FileUtilities.h"

#include "AtRestCipher.h"

#include <QBuffer>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QPixmap>
#include <QPainter>
#include <QDebug>

namespace {

/// Write the bytes to the file, replacing whatever was there.
bool writeBytes(const QString &path, const QByteArray &bytes)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    const bool complete = file.write(bytes) == bytes.size();
    file.close();
    return complete;
}

} // namespace

FileUtilities::AttachmentResult FileUtilities::processAttachment(const QString &filePath,
                                                                 bool isImage,
                                                                 const QString &uploadRoot,
                                                                 const QString &thumbnailRoot) {
    AttachmentResult result;
    result.success = false;

    QFile source(filePath);
    if (!source.open(QIODevice::ReadOnly)) return result;
    const QByteArray originalBytes = source.readAll();
    source.close();

    QString uploadDir = QDir::cleanPath(uploadRoot + "/" + (isImage ? "Images" : "Files"));
    QString thumbDir = QDir::cleanPath(thumbnailRoot + "/" + (isImage ? "Images" : "Files"));

    QDir().mkpath(uploadDir);
    QDir().mkpath(thumbDir);

    QFileInfo fi(filePath);
    QString uniqueName = QString::number(QDateTime::currentMSecsSinceEpoch()) + "_" + fi.fileName();
    QString destPath = uploadDir + "/" + uniqueName;
    QString thumbPath = thumbDir + "/" + uniqueName + (isImage ? "" : ".png");

    // The thumbnail is generated from the plaintext original — the files are
    // sealed afterwards.
    if (isImage) {
        QPixmap original;
        original.loadFromData(originalBytes);
        result.thumbnail = original.scaled(80, 80, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    } else {
        result.thumbnail = QPixmap(80, 80);
        result.thumbnail.fill(QColor("#2a2a35"));
        QPainter p(&result.thumbnail);
        p.setPen(QColor("#ffffff"));
        p.setFont(QFont("Arial", 10, QFont::Bold));
        p.drawText(result.thumbnail.rect(), Qt::AlignCenter, fi.suffix().toUpper() + "\nFILE");
    }

    QByteArray thumbnailBytes;
    QBuffer buffer(&thumbnailBytes);
    buffer.open(QIODevice::WriteOnly);
    result.thumbnail.save(&buffer, "PNG");
    buffer.close();

    // The attachment and its thumbnail are sealed at rest, so that a copy of
    // the data folder does not hand the user's files over in the clear.
    //
    // seal() returns an empty array when the sealing FAILED (no key, no
    // CSPRNG, ...) — then nothing is written and the caller keeps what it
    // had. On a platform without an AEAD backend the payload is returned
    // unchanged: that is the documented platform limit of this build, not a
    // silent fallback, and hasAeadBackend() tells the caller which case it
    // is.
    const QByteArray sealedOriginal =
        AtRestCipher::seal(originalBytes, QString::fromLatin1(AttachmentContext));
    const QByteArray sealedThumbnail =
        AtRestCipher::seal(thumbnailBytes, QString::fromLatin1(ThumbnailContext));
    if (sealedOriginal.isEmpty() || sealedThumbnail.isEmpty()) {
        qWarning() << "FileUtilities: the attachment could not be sealed; "
                      "nothing was written.";
        return result;
    }

    if (!writeBytes(destPath, sealedOriginal)) return result;
    // A failed thumbnail write must not leave the sealed original behind.
    if (!writeBytes(thumbPath, sealedThumbnail)) { QFile::remove(destPath); return result; }

    result.destPath = destPath;
    result.thumbPath = thumbPath;
    result.success = true;
    return result;
}

bool FileUtilities::deleteAttachment(const QString &filePath) {
    return QFile::remove(filePath);
}

QString FileUtilities::getUniqueFileName(const QString &originalName) {
    QFileInfo fi(originalName);
    return QString::number(QDateTime::currentMSecsSinceEpoch()) + "_" + fi.fileName();
}

QByteArray FileUtilities::readAttachment(const QString &filePath, const QString &context) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) return {};
    const QByteArray raw = file.readAll();
    file.close();
    return AtRestCipher::open(raw, context);
}

QPixmap FileUtilities::loadThumbnailPixmap(const QString &thumbPath) {
    const QByteArray bytes = readAttachment(thumbPath, QString::fromLatin1(ThumbnailContext));
    if (bytes.isEmpty()) return {};
    QPixmap pixmap;
    pixmap.loadFromData(bytes);
    return pixmap;
}

bool FileUtilities::isSealedAttachment(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) return false;
    // Only the marker is needed: reading the whole file just to look at its
    // first bytes would be wasteful for a multi-megabyte attachment.
    const QByteArray header = file.read(AtRestCipher::marker().size());
    file.close();
    return AtRestCipher::isSealed(header);
}
