#ifndef THEME_SELECTION_DIALOG_H
#define THEME_SELECTION_DIALOG_H

#include "../../utils/ModernDialog.h"
#include <QGridLayout>
#include <QToolButton>
#include <QLineEdit>
#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QObject>

#include <functional>

class QWidget;
class IUserManager;

class ThemeSelectionDialog : public QWidget {
    Q_OBJECT
public:
    /** username — để save/load accent color per-user */
    /// `users` is injected by the owner (registry-resolved). See AboutDialog.
    explicit ThemeSelectionDialog(const QString &username, IUserManager *users,
                                  QWidget *parent = nullptr);

    /**
     * @brief How the owner applies a background change to the shell.
     *
     * The dialog never touches the shell window itself: the overlay and the
     * event filter that keeps it in sync are owned by the plugin, whose
     * lifetime is the shell session's. The dialog is closed long before the
     * plugin is unloaded, so an event filter installed from here would
     * either die with the dialog (the overlay stops tracking the window) or
     * outlive it with nobody left to remove it (the shell keeps calling into
     * an unloaded DLL).
     */
    using BackgroundRequest =
        std::function<void(const QString &imagePath, int opacity, bool enabled)>;

    void setBackgroundRequest(BackgroundRequest request) {
        m_backgroundRequest = std::move(request);
    }

private:
    void addColorOption(QGridLayout *layout, int row, int col, const QString &name, const QString &color);
    void applyBackground();

    QString     m_selectedColor;
    QString     m_username;
    QString     m_bgImagePath;
    bool        m_useBgImage = false;
    int         m_bgOpacity = 30;   // 0-100

    /// Set by the owner (the plugin): applies the background to the shell.
    BackgroundRequest m_backgroundRequest;

    QCheckBox  *m_useBgCheckbox = nullptr;
    QLineEdit  *m_bgPathEdit   = nullptr;
    QSlider    *m_opacitySlider = nullptr;
    QLabel     *m_opacityLabel  = nullptr;
    QLabel     *m_bgPreview     = nullptr;
    IUserManager *m_users = nullptr;
};

#endif // THEME_SELECTION_DIALOG_H
