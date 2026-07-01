#include "ThemeSelectionDialog.h"
#include "StyleManager.h"
#include "UserManager.h"
#include "UserAuthPlugin.h"
#include "LanguageManager.h"
#include <QLabel>
#include <QPushButton>
#include <QStyle>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QApplication>
#include <QMainWindow>
#include <QPainter>
#include <QCheckBox>
#include <QPointer>

// Helper: find the top-level QMainWindow
static QMainWindow* findMainWindow() {
    for (QWidget *w : QApplication::topLevelWidgets())
        if (auto *mw = qobject_cast<QMainWindow*>(w)) return mw;
    return nullptr;
}

ThemeSelectionDialog::ThemeSelectionDialog(const QString &username, QWidget *parent)
    : QWidget(parent)
    , m_username(username)
{

    // Restore per-user theme on open
    QString savedColor = UserManager::instance()->getUserPref(username, "theme", StyleManager::accentColor());
    if (!savedColor.isEmpty()) StyleManager::setAccentColor(savedColor);

    // Restore background prefs
    m_useBgImage  = UserManager::instance()->getUserPref(username, "use_bg_image", "0") == "1";
    m_bgImagePath = UserManager::instance()->getUserPref(username, "bg_image_path", "");
    m_bgOpacity   = UserManager::instance()->getUserPref(username, "bg_opacity", "30").toInt();

    QVBoxLayout *layout = new QVBoxLayout();
    layout->setContentsMargins(16, 16, 16, 16);
    layout->setSpacing(16);

    // ── Section: Màu giao diện ─────────────────────────────────────────────
    QLabel *desc = new QLabel(UserAuthPlugin::translate("theme.desc"), this);
    desc->setStyleSheet("color: #94a3b8; font-size: 13px;");
    layout->addWidget(desc);

    QGridLayout *grid = new QGridLayout();
    grid->setSpacing(12);
    addColorOption(grid, 0, 0, "Indigo",   "#6366f1");
    addColorOption(grid, 0, 1, "Emerald",  "#10b981");
    addColorOption(grid, 0, 2, "Rose",     "#f43f5e");
    addColorOption(grid, 1, 0, "Amber",    "#f59e0b");
    addColorOption(grid, 1, 1, "Cyan",     "#06b6d4");
    addColorOption(grid, 1, 2, "Violet",   "#8b5cf6");
    layout->addLayout(grid);

    // ── Divider ────────────────────────────────────────────────────────────
    QFrame *line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setStyleSheet("background: rgba(255,255,255,0.08); border: none; max-height: 1px;");
    layout->addWidget(line);

    // ── Section: Ảnh nền ──────────────────────────────────────────────────
    QHBoxLayout *bgHeaderLayout = new QHBoxLayout();
    QLabel *bgTitle = new QLabel("🖼️  " + UserAuthPlugin::translate("theme.bg_image"), this);
    bgTitle->setStyleSheet("color: #e2e8f0; font-size: 12px; font-weight: bold;");
    
    m_useBgCheckbox = new QCheckBox(UserAuthPlugin::translate("theme.use_bg"), this);
    m_useBgCheckbox->setStyleSheet("color: #94a3b8; font-size: 12px;");
    m_useBgCheckbox->setChecked(m_useBgImage);
    
    bgHeaderLayout->addWidget(bgTitle);
    bgHeaderLayout->addStretch();
    bgHeaderLayout->addWidget(m_useBgCheckbox);
    layout->addLayout(bgHeaderLayout);

    // Path row
    QHBoxLayout *pathRow = new QHBoxLayout();
    pathRow->setSpacing(8);
    m_bgPathEdit = new QLineEdit(m_bgImagePath, this);
    m_bgPathEdit->setPlaceholderText(UserAuthPlugin::translate("theme.bg_placeholder"));
    m_bgPathEdit->setReadOnly(true);
    m_bgPathEdit->setFixedHeight(36);
    m_bgPathEdit->setStyleSheet("color: #94a3b8; font-size: 11px;");

    QPushButton *browseBtn = new QPushButton("📂  " + UserAuthPlugin::translate("theme.browse"), this);
    browseBtn->setFixedHeight(36);
    browseBtn->setFixedWidth(110);
    browseBtn->setCursor(Qt::PointingHandCursor);
    connect(browseBtn, &QPushButton::clicked, this, [this]() {
        QString path = QFileDialog::getOpenFileName(
            this, UserAuthPlugin::translate("theme.select_bg"),
            QDir::homePath(),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)");
        if (!path.isEmpty()) {
            m_bgImagePath = path;
            m_bgPathEdit->setText(path);
            // Update preview
            if (m_bgPreview) {
                QPixmap pm(path);
                if (!pm.isNull()) {
                    m_bgPreview->setPixmap(pm.scaled(m_bgPreview->size(),
                        Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
                }
            }
        }
    });

    QPushButton *clearBtn = new QPushButton("✕", this);
    clearBtn->setFixedSize(36, 36);
    clearBtn->setCursor(Qt::PointingHandCursor);
    clearBtn->setToolTip(UserAuthPlugin::translate("theme.clear_bg"));
    connect(clearBtn, &QPushButton::clicked, this, [this]() {
        m_bgImagePath.clear();
        m_bgPathEdit->clear();
        if (m_bgPreview) {
            m_bgPreview->clear();
            m_bgPreview->setText("No image");
        }
    });

    pathRow->addWidget(m_bgPathEdit, 1);
    pathRow->addWidget(browseBtn);
    pathRow->addWidget(clearBtn);
    layout->addLayout(pathRow);

    // Preview + opacity row
    QHBoxLayout *previewRow = new QHBoxLayout();
    previewRow->setSpacing(12);

    // Mini preview
    m_bgPreview = new QLabel(this);
    m_bgPreview->setFixedSize(80, 55);
    m_bgPreview->setAlignment(Qt::AlignCenter);
    m_bgPreview->setStyleSheet("background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; color: #475569; font-size: 10px;");
    m_bgPreview->setText("No image");
    if (!m_bgImagePath.isEmpty()) {
        QPixmap pm(m_bgImagePath);
        if (!pm.isNull())
            m_bgPreview->setPixmap(pm.scaled(m_bgPreview->size(),
                Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
    }

    // Opacity slider
    QWidget *opacityBox = new QWidget(this);
    QVBoxLayout *obv = new QVBoxLayout(opacityBox);
    obv->setContentsMargins(0, 0, 0, 0);
    obv->setSpacing(4);

    QHBoxLayout *opLabelRow = new QHBoxLayout();
    QLabel *opLbl = new QLabel(UserAuthPlugin::translate("theme.opacity"), this);
    opLbl->setStyleSheet("color: #94a3b8; font-size: 11px;");
    m_opacityLabel = new QLabel(QString("%1%").arg(m_bgOpacity), this);
    m_opacityLabel->setStyleSheet("color: #a5b4fc; font-size: 11px; font-weight: bold;");
    opLabelRow->addWidget(opLbl);
    opLabelRow->addStretch();
    opLabelRow->addWidget(m_opacityLabel);

    m_opacitySlider = new QSlider(Qt::Horizontal, this);
    m_opacitySlider->setRange(0, 100);
    m_opacitySlider->setValue(m_bgOpacity);
    m_opacitySlider->setStyleSheet(
        "QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; }"
        "QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; background: #6366f1; border-radius: 8px; }"
        "QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }"
    );
    connect(m_opacitySlider, &QSlider::valueChanged, this, [this](int v) {
        m_bgOpacity = v;
        if (m_opacityLabel) m_opacityLabel->setText(QString("%1%").arg(v));
    });

    obv->addLayout(opLabelRow);
    obv->addWidget(m_opacitySlider);

    previewRow->addWidget(m_bgPreview);
    previewRow->addWidget(opacityBox, 1);
    layout->addLayout(previewRow);

    // -- Collect bg widgets for enable/disable toggling --
    QList<QWidget*> bgWidgets = {m_bgPathEdit, browseBtn, clearBtn, m_bgPreview, opacityBox};
    auto setBgEnabled = [bgWidgets](bool en) {
        for (auto *w : bgWidgets) w->setEnabled(en);
    };
    setBgEnabled(m_useBgImage);  // Set initial state
    connect(m_useBgCheckbox, &QCheckBox::toggled, this, [setBgEnabled](bool checked) {
        setBgEnabled(checked);
    });

    layout->addStretch();

    // ── Buttons ─────────────────────────────────────────────────────────────
    QHBoxLayout *btnLayout = new QHBoxLayout();

    QPushButton *applyBtn = new QPushButton(UserAuthPlugin::translate("theme.apply"), this);
    applyBtn->setObjectName("primary");
    applyBtn->setFixedHeight(42);
    applyBtn->setFixedWidth(140);
    applyBtn->setCursor(Qt::PointingHandCursor);
    applyBtn->setDefault(true);

    btnLayout->addStretch();
    btnLayout->addWidget(applyBtn);
    layout->addLayout(btnLayout);

    connect(applyBtn, &QPushButton::clicked, this, [this]() {
        // Apply accent color
        if (!m_selectedColor.isEmpty()) {
            StyleManager::setAccentColor(m_selectedColor);
            UserManager::instance()->setUserPref(m_username, "theme", m_selectedColor);
        }
        bool useBg = m_useBgCheckbox->isChecked();
        // Toggle glassmorphism based on checkbox
        StyleManager::setGlassmorphismEnabled(useBg);
        // Save background prefs
        UserManager::instance()->setUserPref(m_username, "use_bg_image",  useBg ? "1" : "0");
        UserManager::instance()->setUserPref(m_username, "bg_image_path", m_bgImagePath);
        UserManager::instance()->setUserPref(m_username, "bg_opacity",    QString::number(m_bgOpacity));
        m_useBgImage = useBg;
        applyBackground();
    });
    
    setLayout(layout);
}

void ThemeSelectionDialog::applyBackground() {
    QMainWindow *mw = findMainWindow();
    if (!mw) return;
    applyGlobalBackground(mw, m_bgImagePath, m_bgOpacity, m_useBgImage);
}

void ThemeSelectionDialog::applyGlobalBackground(QMainWindow *mw, const QString &bgPath, int bgOpacity, bool useBg) {
    if (!mw) return;

    if (!useBg || bgPath.isEmpty()) {
        // Remove background overlay
        QLabel *oldOverlay = mw->findChild<QLabel*>("bgGlobalOverlay");
        if (oldOverlay) oldOverlay->deleteLater();
        return;
    }

    QLabel *overlay = mw->findChild<QLabel*>("bgGlobalOverlay");
    if (!overlay) {
        overlay = new QLabel(mw);
        overlay->setObjectName("bgGlobalOverlay");
        // Must be transparent for mouse events so user can click through
        overlay->setAttribute(Qt::WA_TransparentForMouseEvents);
        overlay->setScaledContents(true);
        // Ensure it stays on top
        overlay->raise();
        overlay->show();
        
        // We need the overlay to resize with the main window.
        // A simple event filter can do this.
        class OverlayResizer : public QObject {
        public:
            OverlayResizer(QObject* parent) : QObject(parent) {}
            bool eventFilter(QObject* obj, QEvent* event) override {
                if (event->type() == QEvent::Resize) {
                    QWidget* w = qobject_cast<QWidget*>(obj);
                    if (w) {
                        QLabel* overlay = w->findChild<QLabel*>("bgGlobalOverlay");
                        if (overlay) {
                            overlay->setGeometry(w->rect());
                            overlay->lower(); // Keep it at the bottom
                        }
                    }
                }
                return QObject::eventFilter(obj, event);
            }
        };
        // Avoid adding multiple filters
        if (!mw->property("hasOverlayResizer").toBool()) {
            mw->installEventFilter(new OverlayResizer(mw));
            mw->setProperty("hasOverlayResizer", true);
        }
    }
    
    // Ensure overlay is correctly sized immediately and on bottom
    overlay->setGeometry(mw->rect());
    overlay->lower();

    QPixmap pm(bgPath);
    if (!pm.isNull()) {
        // Apply color tint blend manually to the pixmap
        QPixmap tinted = pm;
        QPainter p(&tinted);
        QColor accent(StyleManager::accentColor());
        double alpha = bgOpacity / 100.0;
        accent.setAlpha(qRound(alpha * 255));
        p.fillRect(tinted.rect(), accent);
        p.end();
        
        overlay->setPixmap(tinted);
        overlay->show();
    } else {
        overlay->hide();
    }
}

void ThemeSelectionDialog::addColorOption(QGridLayout *layout, int row, int col,
                                           const QString &name, const QString &color) {
    QToolButton *btn = new QToolButton(this);
    btn->setObjectName("themeOption");
    btn->setText(name);
    btn->setFixedSize(130, 50);
    btn->setCheckable(true);
    btn->setAutoExclusive(true);
    btn->setCursor(Qt::PointingHandCursor);
    btn->setProperty("colorHex", color);

    QPixmap pix(16, 16);
    pix.fill(QColor(color));
    btn->setIcon(QIcon(pix));
    btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    connect(btn, &QToolButton::clicked, this, [this, btn, color]() {
        m_selectedColor = color;
        for (auto *b : findChildren<QToolButton*>("themeOption")) {
            b->setProperty("selected", b == btn);
            b->style()->unpolish(b);
            b->style()->polish(b);
        }
    });

    layout->addWidget(btn, row, col);

    // Mark current user's saved color as selected
    QString currentAccent = StyleManager::accentColor();
    if (color.toLower() == currentAccent.toLower()) {
        btn->setChecked(true);
        btn->setProperty("selected", true);
        m_selectedColor = color;
    }
}
