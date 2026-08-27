#ifndef FILEUTILITIES_H
#define FILEUTILITIES_H

#include <QString>
#include <QStringList>
#include <QPixmap>
#include "Global.h"

class APP_EXPORT FileUtilities {
public:
    struct AttachmentResult {
        QString destPath;
        QString thumbPath;
        QPixmap thumbnail;
        bool success;
    };

    static AttachmentResult processAttachment(const QString &filePath,
                                              bool isImage,
                                              const QString &uploadRoot,
                                              const QString &thumbnailRoot);
    static bool deleteAttachment(const QString &filePath);
    static QString getUniqueFileName(const QString &originalName);
};

#endif // FILEUTILITIES_H
