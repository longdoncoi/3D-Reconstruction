#include "ShellService.h"
#include "SignalBus.h"

#include <QMenu>
#include <QMainWindow>
#include <QDockWidget>

#include <utility>

ShellService::ShellService(SignalBus* bus)
    : m_bus(bus)
{
}

ShellService::~ShellService()
{
    // Chỉ dọn menu KHÔNG có parent (test hoặc shell chưa kịp gắn parent);
    // menu đã parent cho MainWindow do Qt quản lý.
    for (const QPointer<QMenu>& menu : std::as_const(m_menus)) {
        if (menu && !menu->parent())
            delete menu;
    }
}

QMenu* ShellService::getMenu(const QString& id)
{
    const auto it = m_menus.constFind(id);
    if (it != m_menus.constEnd() && it.value())
        return it.value();

    auto* menu = new QMenu(m_menuParent);
    m_menus.insert(id, menu);
    return menu;
}

QMenuBar* ShellService::menuBar()
{
    return m_menuBar;
}

void ShellService::updateMenuStates()
{
    if (m_bus)
        emit m_bus->stateChanged();
}

QWidget* ShellService::getTabPanel(const QString& tabName)
{
    const auto it = m_tabPanels.constFind(tabName);
    if (it != m_tabPanels.constEnd() && it.value())
        return it.value();
    return nullptr;
}

IProgressDialog* ShellService::getProgressDialog()
{
    return m_progressDialog;
}

void ShellService::registerTabPanel(const QString& key, QWidget* panel)
{
    if (panel)
        m_tabPanels.insert(key, panel);
    else
        m_tabPanels.remove(key);
}

void ShellService::addDockWidget(Qt::DockWidgetArea area, QDockWidget* dockWidget)
{
    if (auto* mw = qobject_cast<QMainWindow*>(m_menuParent.data())) {
        mw->addDockWidget(area, dockWidget);
    }
}

void ShellService::deleteDockWidget(QDockWidget* dockWidget)
{
    if (!dockWidget)
        return;
    if (auto* mw = qobject_cast<QMainWindow*>(m_menuParent.data()))
        mw->removeDockWidget(dockWidget);
    delete dockWidget;
}

QWidget* ShellService::dialogParent()
{
    return m_menuParent.data();
}

QWidget* ShellService::centralWidget()
{
    if (auto* mw = qobject_cast<QMainWindow*>(m_menuParent.data())) {
        return mw->centralWidget();
    }
    return nullptr;
}

void ShellService::setVisible(bool visible)
{
    if (m_menuParent) {
        m_menuParent->setVisible(visible);
    }
}
