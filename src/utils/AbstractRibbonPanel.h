#ifndef ABSTRACT_RIBBON_PANEL_H
#define ABSTRACT_RIBBON_PANEL_H

#include <QColor>
#include <QList>
#include <QPointer>
#include <QString>
#include <QHBoxLayout>
#include <QObject>
#include "Global.h"

class IAppContext;
class QGroupBox;
class QToolButton;
class QHBoxLayout;
class QVBoxLayout;
class QWidget;

/**
 * @brief Shared scaffolding for ribbon panel groups across plugins.
 *
 * All five "RibbonUI" classes previously repeated the same recipe: create a
 * QGroupBox in the panel's QHBoxLayout, an icon-painted tool-button row with
 * ToolButtonTextUnderIcon style, a stretch, and a highlighted title label
 * before the trailing panel spacer. This base encapsulates that recipe so a
 * plugin only declares its buttons.
 *
 * Usage:
 *   if (!beginPanel(parentPanel, "myGroup")) return;
 *   m_btnFoo = addButton("🚀", "#3b82f6", "#2563eb", ctx()->translate("x"), 100);
 *   finishPanel(ctx()->translate("menu.group"));
 * beginPanel()/finishPanel() may be repeated for extra groups in the same tab.
 */
class APP_EXPORT AbstractRibbonPanel : public QObject {
    Q_OBJECT
protected:
    AbstractRibbonPanel(IAppContext *ctx, QObject *parent = nullptr);
    ~AbstractRibbonPanel() override;

    /// Creates a QGroupBox (with button row) inside parentPanel's QHBoxLayout.
    /// @return false when parentPanel has no QHBoxLayout to host the group.
    bool beginPanel(QWidget *parentPanel, const QString &groupObjectName);

    /// Adds an icon-painted tool button to the current group's row.
    /// @param visible pass false to keep runtime-toggled buttons out of view.
    QToolButton *addButton(const QString &emoji, const QString &colorFrom,
                           const QString &colorTo, const QString &translatedText,
                           int minWidth = 80, bool visible = true);

    /// Adds the title label (translated at call time) and inserts the group
    /// before the trailing stretch of the panel layout.
    void finishPanel(const QString &translatedGroupTitle);

    IAppContext *ctx() const { return m_ctx; }
    QGroupBox   *group() const { return m_group.data(); }
    QHBoxLayout *buttonRow() const { return m_row; }

private:
    IAppContext  *m_ctx = nullptr;
    QPointer<QGroupBox> m_group;
    QList<QPointer<QGroupBox>> m_groups;
    QVBoxLayout  *m_vbox = nullptr;
    QHBoxLayout  *m_row = nullptr;
    QHBoxLayout  *m_parentLayout = nullptr;
};

#endif // ABSTRACT_RIBBON_PANEL_H
