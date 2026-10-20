#ifndef SHELLSERVICE_H
#define SHELLSERVICE_H

#include "IShellService.h"
#include "Global.h"
#include "IProgressDialog.h"

#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QPointer>
#include <QWidget>

class SignalBus;

/**
 * @brief Concrete IShellService: menu cache, tab-panel registry and the
 *        shared progress-dialog handle.
 *
 * Tách khỏi AppShell để contract unit-test được trong AppCore; AppShell vẫn
 * là composition root và chỉ delegate. Mọi con trỏ widget đều là QPointer:
 * panel/dialog do MainWindow sở hữu có thể bị hủy trước shell service khi
 * teardown mà không để lại dangling pointer (lookup trả nullptr thay vì UB).
 *
 * Tab panel lookup is key-only (AppConstants::TabIds): no translated-label
 * fallback, so renaming a language cannot change panel routing.
 *
 * Menu được tạo với parent là widget shell (MainWindow) → Qt sở hữu; menu
 * không có parent (dùng trong test) sẽ được service dọn trong destructor.
 */
class APP_EXPORT ShellService : public IShellService {
public:
    explicit ShellService(SignalBus* bus);
    ~ShellService() override;

    // ─── Wiring (AppShell/MainWindow gọi sau khi UI đã dựng) ──────────────────
    void setMenuParent(QWidget* parent) { m_menuParent = parent; }
    void setMenuBar(QMenuBar* bar) { m_menuBar = bar; }
    void setProgressDialog(IProgressDialog* dialog) { m_progressDialog = dialog; }

    // ─── IShellService ────────────────────────────────────────────────────────
    QMenu*    getMenu(const QString& id) override;
    QMenuBar* menuBar() override;
    void      updateMenuStates() override;
    QWidget*  getTabPanel(const QString& tabName) override;
    IProgressDialog* getProgressDialog() override;
    void      registerTabPanel(const QString& key, QWidget* panel) override;
    void      addDockWidget(Qt::DockWidgetArea area, class QDockWidget* dockWidget) override;
    void      deleteDockWidget(class QDockWidget* dockWidget) override;
    QWidget*  dialogParent() override;
    QWidget*  centralWidget() override;
    void      setVisible(bool visible) override;

private:
    SignalBus*        m_bus = nullptr;

    QPointer<QWidget>              m_menuParent;
    QPointer<QMenuBar>             m_menuBar;
    QPointer<IProgressDialog> m_progressDialog;

    QMap<QString, QPointer<QMenu>>   m_menus;
    QMap<QString, QPointer<QWidget>> m_tabPanels;
};

#endif // SHELLSERVICE_H
