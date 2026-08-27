#include "StyleManager.h"
#include <QApplication>
#include <QFont>
#include <QColor>

QString StyleManager::m_accentColor = "#6366f1"; // Default Indigo
bool    StyleManager::m_glassmorphism = false;      // Off by default

void StyleManager::applyTheme(QApplication *app) {
    QFont defaultFont("Segoe UI Variable Text", 10);
    app->setFont(defaultFont);
    app->setStyleSheet(getGlobalStyleSheet());
}

void StyleManager::setGlassmorphismEnabled(bool enabled) {
    m_glassmorphism = enabled;
    if (qApp) qApp->setStyleSheet(getGlobalStyleSheet());
}

void StyleManager::setAccentColor(const QString &colorHex) {
    m_accentColor = colorHex;
    if (qApp) {
        qApp->setStyleSheet(getGlobalStyleSheet());
    }
}

QString StyleManager::getGlobalStyleSheet() {
    QColor accent(m_accentColor);

    auto derive = [&](int s, int l, int a = 255) {
        QColor c = accent.toHsl();
        c.setHsl(c.hslHue(), qBound(0, s, 255), qBound(0, l, 255), a);
        return c.name(a < 255 ? QColor::HexArgb : QColor::HexRgb);
    };

    // Alpha values: full opacity when no bg, semi-transparent (glassmorphism) when bg is active
    const int a1 = m_glassmorphism ? 200 : 255;  // mainBg
    const int a2 = m_glassmorphism ? 220 : 255;  // surfaceBg
    const int a3 = m_glassmorphism ? 100 : 180;  // borderCol
    const int a4 = m_glassmorphism ? 180 : 220;  // hoverBg
    const int a7 = m_glassmorphism ? 230 : 255;  // inputBg
    const int a8 = m_glassmorphism ? 240 : 255;  // vtkBg

    // %1 = mainBg, %2 = surfaceBg, %3 = borderCol, %4 = hoverBg
    // %5 = accentStr, %6 = accentMuted, %7 = inputBg, %8 = vtkBg
    const QString mainBg      = derive(40, 20, a1);
    const QString surfaceBg   = derive(40, 30, a2);
    const QString borderCol   = derive(50, 45, a3);
    const QString hoverBg     = derive(60, 50, a4);
    const QString accentStr   = accent.name();
    const QString accentMuted = derive(80, 60, 45);
    const QString inputBg     = derive(30, 12, a7);
    const QString vtkBg       = derive(25, 10, a8);

    auto a = [&](QString s) {
        s.replace("%1", mainBg);
        s.replace("%2", surfaceBg);
        s.replace("%3", borderCol);
        s.replace("%4", hoverBg);
        s.replace("%5", accentStr);
        s.replace("%6", accentMuted);
        s.replace("%7", inputBg);
        s.replace("%8", vtkBg);
        return s;
    };

    // --- Part 1: Global base, title bars, dock ---
    QString p1 = a(
        "QWidget { background-color: transparent; color: #f1f5f9;"
        " font-family: 'Segoe UI Variable Text','Segoe UI',sans-serif; font-size:10pt; }"
        "QMainWindow, #mainContainer, #dialogContainer"
        " { background-color: %1; color: #f1f5f9; }"
        "QMainWindow > QWidget, QDockWidget > QWidget"
        " { background-color: %1; }"

        // Title bars
        "#modernTitleBar { background-color: %2; border-bottom: 1px solid %3; }"
        "#modernTitleLabel { color: #f1f5f9; font-weight: bold; font-size: 13pt; background: transparent; border: none; }"
        "#modernCloseBtn, #modernControlBtn {"
        "  background-color: transparent; color: #e2e8f0; border: none;"
        "  font-size: 16px; min-width: 32px; min-height: 32px; padding: 0; }"
        "#modernControlBtn:hover { background-color: %4; color: white; }"
        "#modernControlBtn:checked { background-color: %5; color: white; border-radius: 6px; }"
        "#modernCloseBtn:hover   { background-color: #ef4444; color: white; border-radius: 0px; }"

        // Dock widgets
        "QDockWidget { background-color: %1; color: #f1f5f9;"
        "  titlebar-close-icon: none; titlebar-normal-icon: none; }"
        "QDockWidget::title { background-color: %2; padding: 6px 10px;"
        "  border-bottom: 1px solid %3; color: #f1f5f9; }"
        "QDockWidget > QWidget { background-color: %1; }"

        // VTK frame
        "#vtkFrame { background-color: %8; border: 1px solid %3; border-radius: 8px; }"
    );

    // --- Part 2: MenuBar, Menus, Buttons ---
    QString p2 = a(
        "QMenuBar { background-color: %1; border-bottom: 1px solid %3;"
        "  padding: 2px 4px; color: #f1f5f9; }"
        "QMenuBar::item { padding: 5px 12px; background: transparent;"
        "  border-radius: 4px; color: #f1f5f9; }"
        "QMenuBar::item:selected { background-color: %2; color: %5; }"
        "QMenu { background-color: %2; border: 1px solid %5; padding: 4px;"
        "  border-radius: 8px; color: #f1f5f9; }"
        "QMenu::item { padding: 8px 32px 8px 24px; border-radius: 4px;"
        "  margin: 2px 4px; color: #f1f5f9; }"
        "QMenu::item:selected { background-color: %6; color: %5; }"
        "QMenu::separator { height: 1px; background-color: %3; margin: 4px 8px; }"

        // Buttons
        "QPushButton { background-color: %2; border: 1px solid %3; color: #f1f5f9;"
        "  padding: 7px 16px; border-radius: 6px; font-weight: 500; }"
        "QPushButton:hover { background-color: %4; border-color: %5; }"
        "QPushButton:pressed { background-color: %5; }"
        "QPushButton:disabled { color: #475569; border-color: %3; background-color: %1; }"
        "QPushButton#primary, QPushButton[accent=\"true\"]"
        "  { background-color: %5; border: none; color: white; }"
        "QPushButton#primary:hover { background-color: %4; }"
    );

    // --- Part 3: Inputs, Native dialogs, Tabs ---
    QString p3 = a(
        "QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {"
        "  background-color: %7; border: 1px solid %3; border-radius: 6px;"
        "  padding: 7px 10px; color: #f1f5f9; selection-background-color: %5; }"
        "QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus"
        "  { border: 1px solid %5; }"
        "QComboBox::drop-down { border: none; padding-right: 8px; }"
        "QComboBox QAbstractItemView { background-color: %2; color: #f1f5f9;"
        "  selection-background-color: %5; border: 1px solid %3; border-radius: 4px; outline: none; }"

        // Native dialogs
        "QMessageBox { background-color: %1; color: #f1f5f9; }"
        "QMessageBox QLabel { color: #f1f5f9; background-color: transparent; }"
        "QMessageBox QPushButton { min-width: 80px; }"

        // Tabs
        "QTabWidget::pane { border: 1px solid %3; background: %1; border-radius: 8px; }"
        "QTabBar::tab { background: %2; padding: 8px 16px;"
        "  border-top-left-radius: 6px; border-top-right-radius: 6px;"
        "  margin-right: 2px; color: #94a3b8; }"
        "QTabBar::tab:selected { background: %1; color: %5;"
        "  border: 1px solid %3; border-bottom: none; }"
        "QTabBar::tab:hover { color: #f1f5f9; }"
    );

    // --- Part 4: Tables, Lists, Abstract views ---
    QString p4 = a(
        "QTableWidget, QTableView { background-color: %1; gridline-color: %3;"
        "  border: 1px solid %3; alternate-background-color: %7; color: #f1f5f9;"
        "  selection-background-color: %6; selection-color: %5; }"
        "QTableWidget QHeaderView::section, QTableView QHeaderView::section {"
        "  background-color: %2; color: #94a3b8; border: none;"
        "  border-bottom: 1px solid %3; border-right: 1px solid %3; padding: 6px; }"
        "QHeaderView { background-color: %2; }"
        "QTableWidget::item, QTableView::item { padding: 4px; color: #f1f5f9; }"
        "QTableWidget::item:selected, QTableView::item:selected"
        "  { background-color: %6; color: %5; }"
        "QAbstractItemView { background-color: %1; color: #f1f5f9;"
        "  border: 1px solid %3; outline: none; }"
        "QAbstractItemView::item:selected { background-color: %6; color: %5; }"
        "QAbstractItemView::item:hover { background-color: %4; }"

        // List widgets
        "QListWidget, QListView { background-color: %1; border: 1px solid %3;"
        "  border-radius: 6px; outline: none; color: #f1f5f9; }"
        "QListWidget::item, QListView::item { padding: 6px 10px; border-radius: 4px; color: #f1f5f9; }"
        "QListWidget::item:selected, QListView::item:selected { background-color: %6; color: %5; }"
        "QListWidget::item:hover, QListView::item:hover { background-color: %4; }"

        // Tree
        "QTreeView, QTreeWidget { background-color: %1; color: #f1f5f9;"
        "  border: 1px solid %3; alternate-background-color: %7; }"
        "QTreeView::branch { background-color: %1; }"
    );

    // --- Part 5: Scroll areas, Splitters, Misc ---
    QString p5 = a(
        "QScrollArea { background-color: %1; border: none; }"
        "QAbstractScrollArea > QWidget { background-color: %1; }"
        "QAbstractScrollArea::viewport { background-color: %1; }"

        "QScrollBar:vertical { border: none; background: %1; width: 8px; margin: 0; }"
        "QScrollBar::handle:vertical { background: %3; border-radius: 4px; min-height: 30px; }"
        "QScrollBar::handle:vertical:hover { background: %5; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        "QScrollBar:horizontal { border: none; background: %1; height: 8px; margin: 0; }"
        "QScrollBar::handle:horizontal { background: %3; border-radius: 4px; min-width: 30px; }"
        "QScrollBar::handle:horizontal:hover { background: %5; }"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"

        "QSplitter { background-color: %1; }"
        "QSplitter::handle { background-color: %3; }"
        "QSplitter::handle:horizontal { width: 1px; }"
        "QSplitter::handle:vertical   { height: 1px; }"
        "QSplitter::handle:hover { background-color: %5; }"

        "QMainWindow::separator { background-color: %3; width: 1px; height: 1px; }"
        "QMainWindow::separator:hover { background-color: %5; }"

        "QGroupBox { background-color: %1; border: 1px solid %3; border-radius: 8px;"
        "  margin-top: 20px; padding-top: 10px; color: #f1f5f9; font-weight: bold; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 12px;"
        "  padding: 0 6px; color: %5; }"

        "QProgressBar { background-color: %7; border: 1px solid %3; border-radius: 4px;"
        "  text-align: center; color: #f1f5f9; }"
        "QProgressBar::chunk { background-color: %5; border-radius: 4px; }"

        "QCheckBox, QRadioButton { color: #f1f5f9; spacing: 8px; }"
        "QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px;"
        "  border: 1px solid %3; border-radius: 3px; background-color: %7; }"
        "QCheckBox::indicator:checked { background-color: %5; border-color: %5; }"
        "QRadioButton::indicator { border-radius: 8px; }"
        "QRadioButton::indicator:checked { background-color: %5; border-color: %5; }"

        "QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {"
        "  background-color: %7; border: 1px solid %3; border-radius: 6px;"
        "  padding: 6px; color: #f1f5f9; }"

        "QToolBar { background-color: %1; border-bottom: 1px solid %3;"
        "  spacing: 4px; padding: 2px; }"
        "QToolButton { background-color: transparent; border: none; color: #f1f5f9;"
        "  padding: 4px; border-radius: 4px; }"
        "QToolButton:hover { background-color: %4; }"

        "QToolTip { background-color: %2; color: #f1f5f9; border: 1px solid %3;"
        "  border-radius: 4px; padding: 4px 8px; }"
        "QStatusBar { background-color: %1; border-top: 1px solid %3; color: #94a3b8; }"
        "QLabel { color: #f1f5f9; background-color: transparent; }"
    );

    // --- Part 6: App-specific custom widget names ---
    QString p6 = a(
        "#themeOption { background: %2; border: 2px solid %3; border-radius: 8px;"
        "  color: #94a3b8; font-weight: bold; }"
        "#themeOption:hover { border-color: %5; color: white; }"
        "#themeOption[selected=\"true\"] { border-color: white; background: %5; color: white; }"

        "#sessionPanel { background-color: %1; border-right: 1px solid %3; }"
        "#sessionList  { background: transparent; border: none; }"
        "#sessionList::item { padding: 8px 12px; border-radius: 6px; margin: 2px 8px; }"
        "#sessionList::item:selected { background-color: %6; color: %5; }"
        "#chatHistory  { background-color: %1; border: none; }"
        "#attachmentArea { background-color: %1; border-top: 1px solid %3; }"
    );

    // --- Part 7: Tab Bar (compact title-bar integrated tabs) ---
    QString p7 = a(
        // Tab header bar container (inside title bar)
        "#tabHeaderBar { background: transparent; }"

        // Tab header buttons — default (inactive)
        "#tabHeaderBtn {"
        "  background: transparent;"
        "  border: none;"
        "  border-bottom: 2px solid transparent;"
        "  border-radius: 0px;"
        "  color: rgba(241,245,249,0.55);"
        "  font-size: 10pt;"
        "  font-weight: 500;"
        "  padding: 0 14px;"
        "  min-height: 44px;"
        "}"
        "#tabHeaderBtn:hover {"
        "  background: rgba(255,255,255,0.07);"
        "  color: #f1f5f9;"
        "  border-bottom: 2px solid rgba(99,102,241,0.5);"
        "}"
        "#tabHeaderBtn:checked {"
        "  color: %5;"
        "  border-bottom: 2px solid %5;"
        "  background: rgba(99,102,241,0.10);"
        "  font-weight: 600;"
        "}"

        // Tab content strip (tool buttons row)
        "#tabContentStrip {"
        "  background-color: %7;"
        "  border-bottom: 1px solid %3;"
        "}"

        // Ribbon group boxes
        "#tabContentStrip QGroupBox {"
        "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 255, 255, 0.05), stop:1 rgba(255, 255, 255, 0.02));"
        "  border: 1px solid rgba(255, 255, 255, 0.08);"
        "  border-top: 1px solid rgba(255, 255, 255, 0.15);"
        "  border-radius: 8px;"
        "  margin: 2px 4px;"
        "  padding: 0px 4px;"
        "}"

        // Ribbon tool buttons (icon + text under icon)
        "#tabContentStrip QToolButton {"
        "  background: transparent;"
        "  border: 1px solid transparent;"
        "  border-radius: 6px;"
        "  color: #e2e8f0;"
        "  font-size: 8.5pt;"
        "  font-weight: 500;"
        "  padding: 4px 8px;"
        "  margin: 0px;"
        "}"
        "#tabContentStrip QToolButton:hover {"
        "  background: rgba(99, 102, 241, 0.15);"
        "  border: 1px solid rgba(99, 102, 241, 0.3);"
        "  color: #ffffff;"
        "}"
        "#tabContentStrip QToolButton:pressed {"
        "  background: rgba(99, 102, 241, 0.3);"
        "  border: 1px solid %5;"
        "  color: %5;"
        "}"

        // Hide group title label at bottom of ribbon
        "#tabContentStrip QLabel#groupTitleLabel {"
        "  max-height: 0px;"
        "  min-height: 0px;"
        "  height: 0px;"
        "  margin: 0px;"
        "  padding: 0px;"
        "}"

        // Integrated menu bar (inside title bar)
        "#integratedMenuBar { background: transparent; border: none; }"
        "#integratedMenuBar::item { padding: 4px 10px; border-radius: 4px; }"
        "#integratedMenuBar::item:selected { background: rgba(255,255,255,0.1); color: %5; }"

        // Language switcher button
        "#langSwitchBtn {"
        "  border-radius: 4px; border: none; background: transparent;"
        "  padding: 2px 8px; font-size: 13px; color: #f1f5f9;"
        "}"
        "#langSwitchBtn:hover { background: rgba(255,255,255,0.12); }"
        "#langSwitchBtn::menu-indicator { image: none; }"
    );

    return p1 + p2 + p3 + p4 + p5 + p6 + p7;
}
