#ifndef FILEUTILITIES_H
#define FILEUTILITIES_H

#include <QByteArray>
#include <QString>
#include <QStringList>
#include <QPixmap>
#include "Global.h"

class APP_EXPORT FileUtilities {
public:
    /**
     * @brief Associated-data contexts of the at-rest cipher.
     *
     * They are part of the on-disk contract shared with the Python service
     * (ai_assistant.adapters.persistence.storage_crypto): the copied original
     * and its generated thumbnail are sealed under different contexts, so one
     * can never be replayed as the other.
     */
    static constexpr const char *AttachmentContext = "chat.attachment";
    static constexpr const char *ThumbnailContext = "chat.thumbnail";

    struct AttachmentResult {
        QString destPath;
        QString thumbPath;
        QPixmap thumbnail;
        bool success;
    };

    /// Copy the attachment into the upload folder and generate its thumbnail.
    /// Both files are sealed at rest (see AtRestCipher): the copy must not
    /// hand the user's file over in the clear.
    static AttachmentResult processAttachment(const QString &filePath,
                                              bool isImage,
                                              const QString &uploadRoot,
                                              const QString &thumbnailRoot);
    static bool deleteAttachment(const QString &filePath);
    static QString getUniqueFileName(const QString &originalName);

    /// Read an attachment file and return its plaintext: a sealed envelope is
    /// unsealed with the given context, a legacy plaintext file (written
    /// before the envelope existed) is returned unchanged. An empty array
    /// means the file could not be read or unsealed.
    static QByteArray readAttachment(const QString &filePath, const QString &context);

    /// Whether the stored attachment is a sealed envelope.
    ///
    /// A sealed file cannot be handed to an OS handler: the bytes are
    /// ciphertext, so a "file://" link would open garbage. The chat history
    /// uses this to show such an attachment as a name instead of a link.
    static bool isSealedAttachment(const QString &filePath);
};

#endif // FILEUTILITIES_H
