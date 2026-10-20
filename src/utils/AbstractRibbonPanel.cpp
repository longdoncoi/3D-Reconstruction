#include "AbstractRibbonPanel.h"
#include "IAppContext.h"
#include "IconFactory.h"

#include <QGroupBox>
#include <QLabel>
#include <QToolButton>
#include <QVBoxLayout>

#include <utility>

AbstractRibbonPanel::AbstractRibbonPanel(IAppContext *ctx, QObject *parent)
    : QObject(parent), m_ctx(ctx) {}

AbstractRibbonPanel::~AbstractRibbonPanel()
{
    // Groups are parented to the shell's tab panel, not to this QObject.
    // Delete them before the owning plugin DLL is unloaded; otherwise Qt would
    // destroy their (Qt-owned, but still connected) child widgets later, while
    // the panel may already be gone. QPointer tolerates a panel that was
    // destroyed first.
    for (const QPointer<QGroupBox>& group : std::as_const(m_groups)) {
        delete group.data();
    }
}

bool AbstractRibbonPanel::beginPanel(QWidget *parentPanel, const QString &groupObjectName)
{
    auto *layout = qobject_cast<QHBoxLayout *>(parentPanel->layout());
    if (!layout) return false;
    m_parentLayout = layout;

    m_group = new QGroupBox(parentPanel);
    m_groups.append(m_group);
    m_group->setObjectName(groupObjectName);
    m_group->setTitle("");

    m_vbox = new QVBoxLayout(m_group);
    m_vbox->setContentsMargins(4, 4, 4, 4);
    m_vbox->setSpacing(2);

    m_row = new QHBoxLayout();
    m_row->setContentsMargins(0, 0, 0, 0);
    m_row->setSpacing(5);
    m_vbox->addLayout(m_row);
    return true;
}

QToolButton *AbstractRibbonPanel::addButton(const QString &emoji, const QString &colorFrom,
                                            const QString &colorTo, const QString &translatedText,
                                            int minWidth, bool visible)
{
    auto *button = new QToolButton(m_group);
    button->setIcon(IconFactory::createModern(emoji, QColor(colorFrom), QColor(colorTo)));
    button->setText(translatedText);
    button->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    button->setMinimumWidth(minWidth);
    button->setVisible(visible);
    m_row->addWidget(button);
    return button;
}

void AbstractRibbonPanel::finishPanel(const QString &translatedGroupTitle)
{
    m_vbox->addStretch();

    auto *titleLabel = new QLabel(translatedGroupTitle, m_group);
    titleLabel->setObjectName("groupTitleLabel");
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;");
    m_vbox->addWidget(titleLabel);

    if (m_parentLayout)
        m_parentLayout->insertWidget(m_parentLayout->count() - 1, m_group);
}
