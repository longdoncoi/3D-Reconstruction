#ifndef AI_ATTACHMENT_PREVIEW_FACTORY_H
#define AI_ATTACHMENT_PREVIEW_FACTORY_H

#include "FileUtilities.h"

#include <functional>

class QWidget;

namespace AIAttachmentPreviewFactory {

QWidget *create(QWidget *parent,
                const FileUtilities::AttachmentResult &attachment,
                const std::function<void(QWidget *)> &removeCallback);

}

#endif // AI_ATTACHMENT_PREVIEW_FACTORY_H
