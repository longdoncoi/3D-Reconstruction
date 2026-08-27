#include "SettingsDialog.h"
#include "AdminUserManagerDialog.h"
#include "IAIService.h"
#include "IAppContext.h"
#include "IMailService.h"
#include "ISceneService.h"
#include "ISettingsService.h"
#include "ServiceRegistry.h"
#include "SignalBus.h"
#include "ThemeSelectionDialog.h"
#include "UserAuthPlugin.h"
#include "UserManager.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QLinearGradient>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

static QIcon makeNavIcon(const QString &emoji, const QColor &c1,
                         const QColor &c2) {
  QPixmap pix(32, 32);
  pix.fill(Qt::transparent);
  QPainter p(&pix);
  p.setRenderHint(QPainter::Antialiasing);
  QLinearGradient g(0, 0, 0, 32);
  g.setColorAt(0, c1);
  g.setColorAt(1, c2);
  p.setBrush(g);
  p.setPen(Qt::NoPen);
  p.drawRoundedRect(1, 1, 30, 30, 7, 7);
  QFont f = p.font();
  f.setPixelSize(16);
  f.setBold(true);
  p.setFont(f);
  p.setPen(Qt::white);
  p.drawText(QRect(0, 0, 32, 32), Qt::AlignCenter, emoji);
  return QIcon(pix);
}

SettingsDialog::SettingsDialog(const QString &username, bool isAdmin,
                               IAppContext *ctx, QWidget *parent)
    : ModernDialog(UserAuthPlugin::translate("menu.settings"), parent),
      m_username(username), m_isAdmin(isAdmin), m_ctx(ctx) {
  resize(1000, 700);
  setMinimumSize(1000, 700);
  setupUi();
}

void SettingsDialog::setupUi() {
  QWidget *content = new QWidget(this);
  QHBoxLayout *root = new QHBoxLayout(content);
  root->setContentsMargins(0, 0, 0, 0);
  root->setSpacing(0);

  // ── Left navigation list ──────────────────────────────────────────────────
  m_navList = new QListWidget(content);
  m_navList->setObjectName("settingsNavList");
  m_navList->setFixedWidth(200);
  m_navList->setStyleSheet(
      "QListWidget { background: rgba(0,0,0,0.18); border: none; border-right: "
      "1px solid rgba(255,255,255,0.08); border-radius: 0; }"
      "QListWidget::item { padding: 12px 16px; border-radius: 6px; margin: 4px "
      "6px; color: #cbd5e1; font-size: 10pt; }"
      "QListWidget::item:selected { background: rgba(99,102,241,0.25); color: "
      "#a5b4fc; font-weight: 600; }"
      "QListWidget::item:hover { background: rgba(255,255,255,0.07); }");

  // ── Right info panel / Stack ───────────────────────────────────────────────
  m_stack = new QStackedWidget(content);

  QWidget *infoPanel = new QWidget(m_stack);
  QVBoxLayout *infoPl = new QVBoxLayout(infoPanel);
  infoPl->setContentsMargins(32, 32, 32, 32);
  infoPl->setSpacing(16);

  QLabel *hintIcon = new QLabel("⚙️", infoPanel);
  hintIcon->setAlignment(Qt::AlignCenter);
  hintIcon->setStyleSheet("font-size: 48px;");

  QLabel *hintTitle =
      new QLabel(UserAuthPlugin::translate("menu.settings"), infoPanel);
  hintTitle->setAlignment(Qt::AlignCenter);
  hintTitle->setStyleSheet(
      "font-size: 16pt; font-weight: bold; color: #e2e8f0;");

  QLabel *hintDesc =
      new QLabel(UserAuthPlugin::translate("settings.hint_desc"), infoPanel);
  hintDesc->setAlignment(Qt::AlignCenter);
  hintDesc->setWordWrap(true);
  hintDesc->setStyleSheet("color: #64748b; font-size: 10pt;");

  infoPl->addStretch();
  infoPl->addWidget(hintIcon);
  infoPl->addWidget(hintTitle);
  infoPl->addWidget(hintDesc);
  infoPl->addStretch();

  m_stack->addWidget(infoPanel);

  // Create pages
  QWidget *adminPage = nullptr;
  if (m_isAdmin) {
    adminPage = new AdminUserManagerDialog(m_stack);
    m_stack->addWidget(adminPage);
  }

  QWidget *themePage = new ThemeSelectionDialog(m_username, m_stack);
  m_stack->addWidget(themePage);

  auto createTitle = [](const QString &titleText) -> QLabel * {
    QLabel *title = new QLabel(titleText);
    title->setStyleSheet("font-size: 14pt; font-weight: bold; color: #e2e8f0; "
                         "margin-bottom: 10px;");
    return title;
  };

  auto createViewSettings = [this, createTitle]() -> QWidget * {
    QWidget *w = new QWidget(m_stack);
    QVBoxLayout *l = new QVBoxLayout(w);
    l->setContentsMargins(32, 32, 32, 32);
    l->addWidget(createTitle(UserAuthPlugin::translate("menu.view")));

    QFormLayout *form = new QFormLayout();
    form->setVerticalSpacing(15);

    QComboBox *renderQuality = new QComboBox();
    renderQuality->addItems({"Thấp (Nhanh)", "Trung bình", "Cao (Đẹp)"});
    renderQuality->setCurrentIndex(
        UserManager::instance()
            ->getUserPref(m_username, "view_render_quality", "2")
            .toInt());
    connect(renderQuality, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
              UserManager::instance()->setUserPref(
                  m_username, "view_render_quality", QString::number(index));
            });
    form->addRow("Chất lượng Render:", renderQuality);

    QCheckBox *showGrid = new QCheckBox("Hiển thị lưới nền (Grid)");
    showGrid->setChecked(UserManager::instance()->getUserPref(
                             m_username, "view_show_grid", "true") == "true");
    connect(showGrid, &QCheckBox::toggled, this, [this](bool checked) {
      UserManager::instance()->setUserPref(m_username, "view_show_grid",
                                           checked ? "true" : "false");
    });
    form->addRow("", showGrid);

    QCheckBox *showAxes = new QCheckBox("Hiển thị trục toạ độ (Axes)");
    showAxes->setChecked(UserManager::instance()->getUserPref(
                             m_username, "view_show_axes", "true") == "true");
    connect(showAxes, &QCheckBox::toggled, this, [this](bool checked) {
      UserManager::instance()->setUserPref(m_username, "view_show_axes",
                                           checked ? "true" : "false");
    });
    form->addRow("", showAxes);

    QComboBox *bgColor = new QComboBox();
    bgColor->addItems({"Tối (Dark)", "Sáng (Light)", "Xám (Gray)"});
    bgColor->setCurrentIndex(UserManager::instance()
                                 ->getUserPref(m_username, "view_bg_color", "0")
                                 .toInt());
    connect(bgColor, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
              UserManager::instance()->setUserPref(m_username, "view_bg_color",
                                                   QString::number(index));
            });
    form->addRow("Màu nền 3D:", bgColor);

    l->addLayout(form);
    l->addStretch();
    return w;
  };

  auto createReconstructionSettings = [this, createTitle]() -> QWidget * {
    QWidget *w = new QWidget(m_stack);
    QVBoxLayout *l = new QVBoxLayout(w);
    l->setContentsMargins(32, 32, 32, 32);
    l->addWidget(createTitle(UserAuthPlugin::translate("menu.reconstruction")));

    QFormLayout *form = new QFormLayout();
    form->setVerticalSpacing(15);

    QComboBox *algorithm = new QComboBox();
    algorithm->addItems({"Mặc định (SGBM)",
                         "Nâng cao (Neural Radiance Fields - NeRF)",
                         "Gaussian Splatting"});
    algorithm->setCurrentIndex(
        UserManager::instance()
            ->getUserPref(m_username, "recon_algorithm", "0")
            .toInt());
    connect(algorithm, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
              UserManager::instance()->setUserPref(
                  m_username, "recon_algorithm", QString::number(index));
            });
    form->addRow("Thuật toán tái tạo:", algorithm);

    QComboBox *density = new QComboBox();
    density->addItems({"Thấp", "Trung bình", "Cao", "Rất cao"});
    density->setCurrentIndex(UserManager::instance()
                                 ->getUserPref(m_username, "recon_density", "1")
                                 .toInt());
    connect(density, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
              UserManager::instance()->setUserPref(m_username, "recon_density",
                                                   QString::number(index));
            });
    form->addRow("Mật độ điểm ảnh (Point Cloud):", density);

    QCheckBox *autoClean = new QCheckBox("Tự động lọc nhiễu (Noise filter)");
    autoClean->setChecked(
        UserManager::instance()->getUserPref(m_username, "recon_auto_clean",
                                             "true") == "true");
    connect(autoClean, &QCheckBox::toggled, this, [this](bool checked) {
      UserManager::instance()->setUserPref(m_username, "recon_auto_clean",
                                           checked ? "true" : "false");
    });
    form->addRow("", autoClean);

    QSpinBox *maxFeatures = new QSpinBox();
    maxFeatures->setRange(1000, 100000);
    maxFeatures->setValue(
        UserManager::instance()
            ->getUserPref(m_username, "recon_max_features", "10000")
            .toInt());
    connect(maxFeatures, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
              UserManager::instance()->setUserPref(
                  m_username, "recon_max_features", QString::number(value));
            });
    form->addRow("Số đặc trưng tối đa (Features):", maxFeatures);

    l->addLayout(form);
    l->addStretch();
    return w;
  };

  auto createAIToolSettings = [this, createTitle]() -> QWidget * {
    QWidget *w = new QWidget(m_stack);
    QVBoxLayout *l = new QVBoxLayout(w);
    l->setContentsMargins(32, 32, 32, 32);
    l->addWidget(createTitle(UserAuthPlugin::translate("menu.ai_tool")));

    QFormLayout *form = new QFormLayout();
    form->setVerticalSpacing(15);

    QComboBox *modelSelection = new QComboBox();
    modelSelection->addItems(
        {"YOLOv8 - Segmentation", "ResNet50 - Classification", "Custom Model"});
    modelSelection->setCurrentIndex(
        UserManager::instance()
            ->getUserPref(m_username, "aitool_model", "0")
            .toInt());
    connect(modelSelection, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
              UserManager::instance()->setUserPref(m_username, "aitool_model",
                                                   QString::number(index));
            });
    form->addRow("Mô hình phát hiện/phân vùng:", modelSelection);

    QSlider *confidence = new QSlider(Qt::Horizontal);
    confidence->setRange(0, 100);
    confidence->setValue(
        UserManager::instance()
            ->getUserPref(m_username, "aitool_confidence", "75")
            .toInt());
    connect(confidence, &QSlider::valueChanged, this, [this](int value) {
      UserManager::instance()->setUserPref(m_username, "aitool_confidence",
                                           QString::number(value));
    });
    form->addRow("Ngưỡng tự tin (Confidence):", confidence);

    QCheckBox *useGpu = new QCheckBox("Sử dụng phần cứng tăng tốc (GPU/CUDA)");
    useGpu->setChecked(UserManager::instance()->getUserPref(
                           m_username, "aitool_use_gpu", "true") == "true");
    connect(useGpu, &QCheckBox::toggled, this, [this](bool checked) {
      UserManager::instance()->setUserPref(m_username, "aitool_use_gpu",
                                           checked ? "true" : "false");
    });
    form->addRow("", useGpu);

    l->addLayout(form);
    l->addStretch();
    return w;
  };

  auto createAIAssistantSettings = [this, createTitle]() -> QWidget * {
    QWidget *w = new QWidget(m_stack);
    QVBoxLayout *l = new QVBoxLayout(w);
    l->setContentsMargins(32, 32, 32, 32);
    l->addWidget(createTitle(UserAuthPlugin::translate("menu.ai_assistant")));

    QFormLayout *form = new QFormLayout();
    form->setVerticalSpacing(15);

    QComboBox *apiProvider = new QComboBox();
    apiProvider->addItems(
        {"OpenAI (ChatGPT)", "Google (Gemini)", "Local (Ollama)"});
    apiProvider->setCurrentIndex(
        UserManager::instance()
            ->getUserPref(m_username, "ai_provider", "1")
            .toInt());
    connect(apiProvider, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
              UserManager::instance()->setUserPref(m_username, "ai_provider",
                                                   QString::number(index));
            });
    form->addRow("Nền tảng AI:", apiProvider);

    QLineEdit *apiKey = new QLineEdit();
    apiKey->setEchoMode(QLineEdit::Password);
    apiKey->setPlaceholderText("Nhập API Key...");
    apiKey->setText(
        UserManager::instance()->getUserPref(m_username, "ai_api_key", ""));
    connect(apiKey, &QLineEdit::textChanged, this, [this](const QString &text) {
      UserManager::instance()->setUserPref(m_username, "ai_api_key", text);
    });
    form->addRow("API Key:", apiKey);

    QDoubleSpinBox *temperature = new QDoubleSpinBox();
    temperature->setRange(0.0, 2.0);
    temperature->setSingleStep(0.1);
    temperature->setValue(UserManager::instance()
                              ->getUserPref(m_username, "ai_temperature", "0.7")
                              .toDouble());
    connect(temperature, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double value) {
              UserManager::instance()->setUserPref(m_username, "ai_temperature",
                                                   QString::number(value));
            });
    form->addRow("Độ sáng tạo (Temperature):", temperature);

    QCheckBox *saveHistory = new QCheckBox("Lưu lịch sử trò chuyện cục bộ");
    saveHistory->setChecked(
        UserManager::instance()->getUserPref(m_username, "ai_save_history",
                                             "true") == "true");
    connect(saveHistory, &QCheckBox::toggled, this, [this](bool checked) {
      UserManager::instance()->setUserPref(m_username, "ai_save_history",
                                           checked ? "true" : "false");
    });
    form->addRow("", saveHistory);

    l->addLayout(form);
    l->addStretch();
    return w;
  };

  auto createMailSettings = [this, createTitle]() -> QWidget * {
    QWidget *w = new QWidget(m_stack);
    QVBoxLayout *l = new QVBoxLayout(w);
    l->setContentsMargins(32, 32, 32, 32);
    l->addWidget(createTitle(UserAuthPlugin::translate("menu.mail")));

    QFormLayout *form = new QFormLayout();
    form->setVerticalSpacing(15);

    QLineEdit *smtpServer = new QLineEdit();
    smtpServer->setText(UserManager::instance()->getUserPref(
        m_username, "mail_smtp_server", "smtp.gmail.com"));
    connect(smtpServer, &QLineEdit::textChanged, this,
            [this](const QString &text) {
              UserManager::instance()->setUserPref(m_username,
                                                   "mail_smtp_server", text);
            });
    form->addRow("Máy chủ SMTP:", smtpServer);

    QSpinBox *port = new QSpinBox();
    port->setRange(1, 65535);
    port->setValue(UserManager::instance()
                       ->getUserPref(m_username, "mail_port", "587")
                       .toInt());
    connect(port, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
              UserManager::instance()->setUserPref(m_username, "mail_port",
                                                   QString::number(value));
            });
    form->addRow("Cổng (Port):", port);

    QLineEdit *email = new QLineEdit();
    email->setPlaceholderText("example@gmail.com");
    email->setText(
        UserManager::instance()->getUserPref(m_username, "mail_email", ""));
    connect(email, &QLineEdit::textChanged, this, [this](const QString &text) {
      UserManager::instance()->setUserPref(m_username, "mail_email", text);
    });
    form->addRow("Địa chỉ Email:", email);

    QLineEdit *password = new QLineEdit();
    password->setEchoMode(QLineEdit::Password);
    password->setPlaceholderText("Mật khẩu ứng dụng");
    password->setText(
        UserManager::instance()->getUserPref(m_username, "mail_password", ""));
    connect(password, &QLineEdit::textChanged, this,
            [this](const QString &text) {
              UserManager::instance()->setUserPref(m_username, "mail_password",
                                                   text);
            });
    form->addRow("Mật khẩu:", password);

    QPushButton *testBtn = new QPushButton("Kiểm tra kết nối");
    testBtn->setFixedWidth(160);
    connect(testBtn, &QPushButton::clicked, this,
            [this, smtpServer, port, email, password]() {
              // Lưu giá trị hiện tại từ UI vào UserManager trước khi test
              auto *um = UserManager::instance();
              if (um) {
                um->setUserPref(m_username, "mail_smtp_server",
                                smtpServer->text());
                um->setUserPref(m_username, "mail_port",
                                QString::number(port->value()));
                um->setUserPref(m_username, "mail_email", email->text());
                um->setUserPref(m_username, "mail_password", password->text());
              }

              // Đồng bộ vào MailService và test thực sự
              if (m_ctx && m_ctx->services()) {
                if (auto *mailSvc = m_ctx->services()->get<IMailService>()) {
                  mailSvc->setCredentials(email->text(), password->text(),
                                          m_username);
                  QString errMsg;
                  bool ok = mailSvc->testConnection(errMsg);
                  if (ok) {
                    QMessageBox::information(
                        this, "Kiểm tra kết nối",
                        "✅ Kết nối SMTP/IMAP thành công!\nEmail: " +
                            email->text());
                  } else {
                    QMessageBox::warning(this, "Kiểm tra kết nối",
                                         "❌ Kết nối thất bại:\n" + errMsg);
                  }
                  return;
                }
              }
              QMessageBox::information(this, "Kiểm tra kết nối",
                                       "Mail service chưa được khởi tạo.");
            });
    form->addRow("", testBtn);

    l->addLayout(form);
    l->addStretch();
    return w;
  };

  QWidget *viewPage = createViewSettings();
  QWidget *reconstructPage = createReconstructionSettings();
  QWidget *aiToolPage = createAIToolSettings();
  QWidget *aiAssistantPage = createAIAssistantSettings();
  QWidget *mailPage = createMailSettings();

  m_stack->addWidget(viewPage);
  m_stack->addWidget(reconstructPage);
  m_stack->addWidget(aiToolPage);
  m_stack->addWidget(aiAssistantPage);
  m_stack->addWidget(mailPage);

  // --- Nav items ---
  if (m_isAdmin) {
    auto *navItem0 = new QListWidgetItem(
        makeNavIcon("👥", QColor("#6366f1"), QColor("#4f46e5")),
        UserAuthPlugin::translate("menu.user_mgmt"), m_navList);
    navItem0->setSizeHint(QSize(0, 48));
  }
  auto *navItem = new QListWidgetItem(
      makeNavIcon("🎨", QColor("#ec4899"), QColor("#db2777")),
      UserAuthPlugin::translate("menu.theme"), m_navList);
  navItem->setSizeHint(QSize(0, 48));

  navItem = new QListWidgetItem(
      makeNavIcon("👁", QColor("#f04d4d"), QColor("#9c1717")),
      UserAuthPlugin::translate("menu.view"), m_navList);
  navItem->setSizeHint(QSize(0, 48));

  navItem = new QListWidgetItem(
      makeNavIcon("🏗", QColor("#d1bc43"), QColor("#9c8817")),
      UserAuthPlugin::translate("menu.reconstruction"), m_navList);
  navItem->setSizeHint(QSize(0, 48));

  navItem = new QListWidgetItem(
      makeNavIcon("🪄", QColor("#64d143"), QColor("#39a11a")),
      UserAuthPlugin::translate("menu.ai_tool"), m_navList);
  navItem->setSizeHint(QSize(0, 48));

  navItem = new QListWidgetItem(
      makeNavIcon("🤖", QColor("#42c8d4"), QColor("#1a96a1")),
      UserAuthPlugin::translate("menu.ai_assistant"), m_navList);
  navItem->setSizeHint(QSize(0, 48));

  navItem = new QListWidgetItem(
      makeNavIcon("📩", QColor("#8d42d4"), QColor("#52138f")),
      UserAuthPlugin::translate("menu.mail"), m_navList);
  navItem->setSizeHint(QSize(0, 48));

  // ── Layout assembly ───────────────────────────────────────────────────────
  root->addWidget(m_navList);
  root->addWidget(m_stack, 1);

  connect(m_navList, &QListWidget::itemClicked, this,
          [=](QListWidgetItem *item) {
            int row = m_navList->row(item);
            int adminOffset = m_isAdmin ? 1 : 0;

            if (m_isAdmin && row == 0) {
              m_stack->setCurrentWidget(adminPage);
            } else if (row == adminOffset + 0) {
              m_stack->setCurrentWidget(themePage);
            } else if (row == adminOffset + 1) {
              m_stack->setCurrentWidget(viewPage);
            } else if (row == adminOffset + 2) {
              m_stack->setCurrentWidget(reconstructPage);
            } else if (row == adminOffset + 3) {
              m_stack->setCurrentWidget(aiToolPage);
            } else if (row == adminOffset + 4) {
              m_stack->setCurrentWidget(aiAssistantPage);
            } else if (row == adminOffset + 5) {
              m_stack->setCurrentWidget(mailPage);
            }
          });

  // Bottom close button
  QWidget *wrapper = new QWidget(this);
  QVBoxLayout *wl = new QVBoxLayout(wrapper);
  wl->setContentsMargins(0, 0, 0, 0);
  wl->setSpacing(0);
  wl->addWidget(content, 1);

  QWidget *footer = new QWidget(wrapper);
  footer->setObjectName("settingsFooter");
  footer->setStyleSheet("#settingsFooter { border-top: 1px solid "
                        "rgba(255,255,255,0.08); background: transparent; }");
  QHBoxLayout *fl = new QHBoxLayout(footer);
  fl->setContentsMargins(16, 10, 16, 10);

  QPushButton *applyBtn =
      new QPushButton(UserAuthPlugin::translate("common.apply"), footer);
  applyBtn->setObjectName("primary");
  applyBtn->setFixedHeight(38);
  applyBtn->setFixedWidth(100);
  applyBtn->setCursor(Qt::PointingHandCursor);

  connect(applyBtn, &QPushButton::clicked, this, [this]() {
    auto *um = UserManager::instance();

    // ── TAB "XEM": Áp dụng màu nền, trục toạ độ, lưới nền ──────────────
    if (m_ctx && m_ctx->scene()) {
      m_ctx->scene()->applyViewSettings(m_username);
    }

    // ── TAB "CÔNG CỤ AI": Áp dụng ngưỡng tin cậy ───────────────────────────
    if (m_ctx && m_ctx->services()) {
      if (auto *aiSvc = m_ctx->services()->get<IAIService>()) {
        int confidenceInt =
            um ? um->getUserPref(m_username, "aitool_confidence", "75").toInt()
               : 75;
        if (confidenceInt <= 0)
          confidenceInt = 75; // fallback
        aiSvc->setConfidenceThreshold(confidenceInt / 100.0f);
      }
    }

    // ── TAB "EMAIL": Đồng bộ thông số vào MailService ─────────────────────
    if (m_ctx && m_ctx->services()) {
      if (auto *mailSvc = m_ctx->services()->get<IMailService>()) {
        QString email =
            um ? um->getUserPref(m_username, "mail_email", "") : QString();
        QString password =
            um ? um->getUserPref(m_username, "mail_password", "") : QString();
        QString dispName =
            um ? um->getUserPref(m_username, "mail_display_name", m_username)
               : m_username;
        mailSvc->setCredentials(email, password, dispName);
      }
    }

    // ── THÔNG BÁO thông qua SignalBus để các plugin khác cập nhật ─────────
    if (m_ctx && m_ctx->signalBus()) {
      emit m_ctx->signalBus()->stateChanged();
    }
  });

  QPushButton *closeBtn =
      new QPushButton(UserAuthPlugin::translate("common.close"), footer);
  closeBtn->setFixedHeight(38);
  closeBtn->setFixedWidth(100);
  closeBtn->setCursor(Qt::PointingHandCursor);
  connect(closeBtn, &QPushButton::clicked, this, &QDialog::accept);

  fl->addStretch();
  fl->addWidget(applyBtn);
  fl->addWidget(closeBtn);
  wl->addWidget(footer);

  setContentLayout(wl);
}
