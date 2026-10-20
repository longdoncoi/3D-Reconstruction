#include "MainWindow.h"
#include "AppConstants.h"
#include "AppShell.h"
#include "CustomProgressDialog.h"
#include "ILanguageService.h"
#include "PluginHost.h"
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
#include <QPushButton>
#include <QStackedWidget>
#include <QScreen>
#include <QGuiApplication>

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

  // 4b. Plugin lifecycle host (AppCore) — loaded after the event loop starts.
  m_pluginHost = std::make_unique<PluginHost>();

  // 5. Build UI (title bar, tabs, vtk frame)
  setupTitleBar(central, rootLayout);
  setupTabContent(central, rootLayout);
  setupVtkFrame(central, rootLayout);

  // 6. Init all services (phải sau khi widget đã được tạo)
  m_shell->initializeServices();

  // 7. Progress dialog (owned by MainWindow, referenced by AppShell)
  auto *dlg = new CustomProgressDialog(
      context()->services()->getRequired<ILanguageService>(), this);
  m_shell->setProgressDialog(dlg);

  // 8. Language service (theme was applied once in main() before the shell was built)

  // 9. Connect language service (interface from the registry)
  auto *langSvc = context()->services()->getRequired<ILanguageService>();
  connect(langSvc, &ILanguageService::languageChanged, this,
          &MainWindow::onLanguageChanged);
  connect(langSvc, &ILanguageService::languageChanged, m_shell->signalBus(),
          &SignalBus::languageChanged);

  // 10. Load plugins DEFERRED — window shows first, plugins load after event loop starts
  //     This dramatically improves perceived startup time.
  QTimer::singleShot(0, this, &MainWindow::loadPlugins);
}

MainWindow::~MainWindow() {
  // Fallback khi cửa sổ bị destroy mà không qua closeEvent (ví dụ qApp->quit()
  // giữa chừng): plugins vẫn phải được cho cơ hội dọn dẹp.
  shutdownPlugins();
  // Destroy the service graph while the widget tree is still alive, instead
  // of relying on QObject's child-deletion order (which runs only in the
  // QMainWindow base destructor, after this class's members are gone).
  // Plugins have already been shut down and their docks deleted, so no
  // consumer can call into these services from a destructor.
  m_shell.reset();
  delete ui;
}

// ─── UI Construction
// ──────────────────────────────────────────────────────────
void MainWindow::setupTitleBar(QWidget * /*root*/,
                               QVBoxLayout * /*rootLayout*/) {
  // Tab definitions: AppConstants::TabIds is the single source of truth for
  // ids and display order (the id doubles as the translation key).
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

  for (const char *tabId : AppConstants::TabIds::BuildOrder) {
    const QString key = QString::fromLatin1(tabId);
    auto *btn = new QPushButton(context()->translate(key), m_tabBarWidget);
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
    m_shell->shell()->registerTabPanel(key, panel);

    connect(btn, &QPushButton::clicked, this,
            [this, key]() { activateTab(key); });

    m_tabBarLayout->addWidget(btn);
  }
  titleLayout->addWidget(m_tabBarWidget);

  // Integrated MenuBar
  auto *menuBar = new QMenuBar(titleBar);
  menuBar->setObjectName("integratedMenuBar");
  menuBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  menuBar->setMinimumHeight(32);
  QAction *dummyAct = menuBar->addAction("");
  dummyAct->setVisible(false);
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

  // Activate first tab (TabIds::BuildOrder is never empty)
  activateTab(QString::fromLatin1(AppConstants::TabIds::BuildOrder[0]));
}

void MainWindow::setupTabContent(QWidget *root, QVBoxLayout *rootLayout) {
  m_tabContent = new QStackedWidget(root);
  m_tabContent->setObjectName("tabContentStrip");
  m_tabContent->setFixedHeight(66);

  for (const char *tabId : AppConstants::TabIds::BuildOrder)
    m_tabContent->addWidget(
        m_tabPanels.value(QString::fromLatin1(tabId)));

  if (!m_activeTab.isEmpty())
    activateTab(m_activeTab);

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
    child->setText(context()->translate("app.title"));
  for (auto it = m_tabButtons.begin(); it != m_tabButtons.end(); ++it)
    it.value()->setText(context()->translate(it.key()));
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
QString MainWindow::resolvePluginsDirectory() const {
  QDir pluginsDir(qApp->applicationDirPath());

  if (pluginsDir.exists("plugins")) {
    pluginsDir.cd("plugins");
  } else {
#if defined(Q_OS_WIN)
    const QString dirName = pluginsDir.dirName();
    if (dirName.toLower() == "debug" || dirName.toLower() == "release") {
      // MSBuild puts plugins in build/plugins/Release
      QDir buildDir = pluginsDir;
      buildDir.cdUp();
      if (buildDir.exists("plugins/" + dirName)) {
        pluginsDir = buildDir;
        pluginsDir.cd("plugins");
        pluginsDir.cd(dirName);
      } else {
        pluginsDir.cdUp();
        if (!pluginsDir.cd("plugins")) {
          qWarning() << "Could not find plugins directory:" << pluginsDir.absolutePath();
          return QString();
        }
      }
    } else {
      qWarning() << "Could not find plugins directory:" << pluginsDir.absolutePath();
      return QString();
    }
#else
    qWarning() << "Could not find plugins directory:" << pluginsDir.absolutePath();
    return QString();
#endif
  }
  return pluginsDir.absolutePath();
}

void MainWindow::loadPlugins() {
  const QString directory = resolvePluginsDirectory();
  if (directory.isEmpty()) return;

  m_pluginHost->discover(directory);
  for (const QString &error : m_pluginHost->takeErrors()) qWarning().noquote() << error;

  if (!m_pluginHost->resolveOrder()) {
    for (const QString &error : m_pluginHost->takeErrors()) qWarning().noquote() << error;
    qWarning() << "Plugin dependency resolution failed — aborting plugin loading.";
    return;
  }

  // Phase-2 services (reconstruction/AI/mail) must be fully registered before
  // any plugin's initialize() runs — explicit call, no event-queue ordering
  // assumption.
  m_shell->initializeDeferredServices();

  if (!m_pluginHost->initializeAll(context())) {
    for (const QString &error : m_pluginHost->takeErrors()) qWarning().noquote() << error;
    close();
  }
}

void MainWindow::shutdownPlugins() {
  if (!m_pluginHost || m_pluginHost->isEmpty()) return;

  // LIFO cleanup (PluginHost) rồi mới ngừng expose services: sau khi mọi
  // plugin ngừng dùng services thì registry ngừng expose them, bất kỳ get()
  // nào trễ hơn sẽ trả nullptr thay vì con trỏ đã chết.
  m_pluginHost->shutdown();
  context()->services()->clear();
}

void MainWindow::closeEvent(QCloseEvent *event) {
  shutdownPlugins();
  event->accept();
}
