#include "MainWindow.h"
#include "AppShell.h"
#include "CustomProgressDialog.h"
#include "IPlugin.h"
#include "LanguageManager.h"
#include "StyleManager.h"
#include "ui_MainWindow.h"

#include <QBoxLayout>
#include <QDebug>
#include <QDir>
#include <QFrame>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMouseEvent>
#include <QPixmap>
#include <QPluginLoader>
#include <QPushButton>
#include <QStackedWidget>
#include <QScreen>
#include <QGuiApplication>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);

  // 1. Frameless window
  setWindowFlags(Qt::Window | Qt::FramelessWindowHint);
  setAttribute(Qt::WA_TranslucentBackground);

  // Prevent frameless window from expanding beyond screen height when docks are opened
  if (QScreen *screen = QGuiApplication::primaryScreen()) {
      setMaximumSize(screen->availableGeometry().size());
      connect(screen, &QScreen::availableGeometryChanged, this, [this](const QRect &geom) {
          setMaximumSize(geom.size());
      });
  }

  // 2. Root container
  QWidget *central = new QWidget(this);
  central->setObjectName("mainContainer");
  setCentralWidget(central);

  QVBoxLayout *rootLayout = new QVBoxLayout(central);
  rootLayout->setContentsMargins(10, 0, 10, 0);
  rootLayout->setSpacing(0);

  // 3. VTK widget — phải tạo sớm vì AppShell cần nó
  m_vtkWidget = new QVTKOpenGLNativeWidget(this);

  // 4. Tạo AppShell (IAppContext implementation)
  m_shell = std::make_unique<AppShell>(this, m_vtkWidget, this);

  // 5. Build UI (title bar, tabs, vtk frame)
  setupTitleBar(central, rootLayout);
  setupTabContent(central, rootLayout);
  setupVtkFrame(central, rootLayout);

  // 6. Init all services (phải sau khi widget đã được tạo)
  m_shell->initializeServices();

  // 7. Progress dialog (owned by MainWindow, referenced by AppShell)
  auto *dlg = new CustomProgressDialog(this);
  m_shell->setProgressDialog(dlg);

  // 8. Apply theme
  StyleManager::applyTheme(qApp);

  // 9. Connect language manager
  connect(&LM, &LanguageManager::languageChanged, this,
          &MainWindow::onLanguageChanged);
  connect(&LM, &LanguageManager::languageChanged, m_shell->signalBus(),
          &SignalBus::languageChanged);

  // 10. Load plugins (sau khi shell + services sẵn sàng)
  loadPlugins();
}

MainWindow::~MainWindow() { delete ui; }

// ─── UI Construction
// ──────────────────────────────────────────────────────────
void MainWindow::setupTitleBar(QWidget * /*root*/,
                               QVBoxLayout * /*rootLayout*/) {
  // Tab definitions
  QStringList tabKeys = {"tab.view", "tab.reconstruction", "tab.ai",
                         "tab.ai_assistant", "tab.mail", "tab.help"};
  // QStringList tabKeys = {
  //     "tab.view"
  // };

  QWidget *titleBar = new QWidget(this);
  titleBar->setObjectName("modernTitleBar");
  titleBar->setFixedHeight(44);

  QHBoxLayout *titleLayout = new QHBoxLayout(titleBar);
  titleLayout->setContentsMargins(10, 0, 0, 0);
  titleLayout->setSpacing(0);

  // Tab bar widget
  m_tabBarWidget = new QWidget(titleBar);
  m_tabBarWidget->setObjectName("tabHeaderBar");
  m_tabBarLayout = new QHBoxLayout(m_tabBarWidget);
  m_tabBarLayout->setContentsMargins(4, 0, 4, 0);
  m_tabBarLayout->setSpacing(2);

  for (const QString &key : tabKeys) {
    auto *btn = new QPushButton(LM_TR(key), m_tabBarWidget);
    btn->setObjectName("tabHeaderBtn");
    btn->setCheckable(true);
    btn->setAutoExclusive(true);
    btn->setCursor(Qt::PointingHandCursor);
    btn->setProperty("tabKey", key);

    // Panel (plugins fill this)
    QWidget *panel = new QWidget();
    panel->setObjectName("tabPanel_" + key);
    QHBoxLayout *pLayout = new QHBoxLayout(panel);
    pLayout->setContentsMargins(4, 2, 4, 2);
    pLayout->setSpacing(4);
    pLayout->addStretch();

    m_tabButtons[key] = btn;
    m_tabPanels[key] = panel;

    // Register panel in shell so plugins can find it
    m_shell->registerTabPanel(key, panel);

    connect(btn, &QPushButton::clicked, this,
            [this, key]() { activateTab(key); });

    m_tabBarLayout->addWidget(btn);
  }
  titleLayout->addWidget(m_tabBarWidget);
  titleLayout->addStretch(1);

  // Integrated MenuBar
  auto *menuBar = new QMenuBar(titleBar);
  menuBar->setObjectName("integratedMenuBar");
  menuBar->setStyleSheet("QMenuBar { background: transparent; border: none; "
                         "padding: 0; color: #f1f5f9; }"
                         "QMenuBar::item { padding: 4px 10px; border-radius: "
                         "4px; background: transparent; }"
                         "QMenuBar::item:selected { background: "
                         "rgba(255,255,255,0.1); color: #6366f1; }");
  titleLayout->addWidget(menuBar);
  m_shell->setMenuBar(menuBar);

  // Window control buttons
  auto makeBtn = [&](const QString &txt, const QString &obj) {
    auto *b = new QPushButton(txt, titleBar);
    b->setObjectName(obj);
    b->setFixedSize(42, 44);
    b->setFlat(true);
    b->setCursor(Qt::PointingHandCursor);
    return b;
  };

  auto *minBtn = makeBtn("−", "modernControlBtn");
  auto *maxBtn = makeBtn("▢", "modernControlBtn");
  auto *closeBtn = makeBtn("✕", "modernCloseBtn");

  titleLayout->addWidget(minBtn);
  titleLayout->addWidget(maxBtn);
  titleLayout->addWidget(closeBtn);

  setMenuWidget(titleBar);
  m_titleBar = titleBar;

  connect(minBtn, &QPushButton::clicked, this, &QMainWindow::showMinimized);
  connect(maxBtn, &QPushButton::clicked, this, [this]() {
      if (this->property("customMaximized").toBool()) {
          this->setGeometry(this->property("normalGeometry").toRect());
          this->setProperty("customMaximized", false);
      } else {
          this->setProperty("normalGeometry", this->geometry());
          this->setGeometry(QGuiApplication::primaryScreen()->availableGeometry());
          this->setProperty("customMaximized", true);
      }
  });
  connect(closeBtn, &QPushButton::clicked, this, &QMainWindow::close);

  // Activate first tab
  if (!tabKeys.isEmpty())
    activateTab(tabKeys.first());
}

void MainWindow::setupTabContent(QWidget *root, QVBoxLayout *rootLayout) {
  m_tabContent = new QStackedWidget(root);
  m_tabContent->setObjectName("tabContentStrip");
  m_tabContent->setFixedHeight(66);

  for (auto it = m_tabPanels.begin(); it != m_tabPanels.end(); ++it)
    m_tabContent->addWidget(it.value());

  rootLayout->addWidget(m_tabContent);
}

void MainWindow::setupVtkFrame(QWidget *root, QVBoxLayout *rootLayout) {
  QWidget *contentArea = new QWidget(root);
  QVBoxLayout *mainLayout = new QVBoxLayout(contentArea);
  mainLayout->setContentsMargins(0, 0, 0, 10);
  mainLayout->setSpacing(0);

  QFrame *vtkFrame = new QFrame(contentArea);
  vtkFrame->setObjectName("vtkFrame");
  vtkFrame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  QVBoxLayout *vtkLayout = new QVBoxLayout(vtkFrame);
  vtkLayout->setContentsMargins(0, 0, 0, 0);

  m_vtkWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  vtkLayout->addWidget(m_vtkWidget);

  mainLayout->addWidget(vtkFrame);
  rootLayout->addWidget(contentArea, 1);
}

void MainWindow::activateTab(const QString &tabName) {
  m_activeTab = tabName;
  if (m_tabPanels.contains(tabName) && m_tabContent)
    m_tabContent->setCurrentWidget(m_tabPanels[tabName]);
  if (m_tabButtons.contains(tabName))
    m_tabButtons[tabName]->setChecked(true);
}

// ─── Language
// ─────────────────────────────────────────────────────────────────
void MainWindow::onLanguageChanged(const QString & /*lang*/) {
  for (auto *child : m_titleBar->findChildren<QLabel *>("modernTitleLabel"))
    child->setText(LM_TR("app.title"));
  for (auto it = m_tabButtons.begin(); it != m_tabButtons.end(); ++it)
    it.value()->setText(LM_TR(it.key()));
}

// ─── Mouse drag (frameless)
// ───────────────────────────────────────────────────
void MainWindow::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton &&
      m_titleBar->rect().contains(
          m_titleBar->mapFromGlobal(event->globalPosition().toPoint()))) {
    m_dragging = true;
    m_dragPos = event->globalPosition().toPoint() - frameGeometry().topLeft();
    event->accept();
  }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event) {
  if (m_dragging && (event->buttons() & Qt::LeftButton)) {
    move(event->globalPosition().toPoint() - m_dragPos);
    event->accept();
  }
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event) {
  m_dragging = false;
  event->accept();
}

// ─── Plugin loading
// ───────────────────────────────────────────────────────────
void MainWindow::loadPlugins() {
  discoverPlugins();
  sortPlugins();
  initPlugins();
}

void MainWindow::discoverPlugins() {
  QDir pluginsDir(qApp->applicationDirPath());

#if defined(Q_OS_WIN)
  const QString dirName = pluginsDir.dirName().toLower();
  if (dirName == "debug" || dirName == "release")
    pluginsDir.cdUp();
#endif

  if (!pluginsDir.cd("plugins")) {
    qWarning() << "Could not find plugins directory:"
               << pluginsDir.absolutePath();
    return;
  }

  for (const QString &fileName : pluginsDir.entryList(QDir::Files)) {
    if (!QLibrary::isLibrary(fileName))
      continue;

    QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
    QObject *obj = loader.instance();
    if (obj) {
      if (auto *ipl = qobject_cast<IPlugin *>(obj)) {
        qDebug() << "Loaded plugin:" << ipl->pluginName();
        m_plugins.append(ipl);
      } else {
        qWarning() << "Plugin" << fileName << "does not implement IPlugin.";
      }
    } else {
      qWarning() << "Failed to load plugin" << fileName << ":"
                 << loader.errorString();
    }
  }
}

void MainWindow::sortPlugins() {
  std::sort(m_plugins.begin(), m_plugins.end(), [](IPlugin *a, IPlugin *b) {
    return a->loadOrder() < b->loadOrder();
  });
}

void MainWindow::initPlugins() {
  // context() returns the AppShell which implements IAppContext
  IAppContext *ctx = context();
  for (IPlugin *plugin : m_plugins)
    plugin->initialize(ctx);
  for (IPlugin *plugin : m_plugins)
    plugin->onAppReady();
}
