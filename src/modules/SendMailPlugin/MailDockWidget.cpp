#include "MailDockWidget.h"

#include "IAppContext.h"
#include "CustomProgressDialog.h"
#include "ModernMessageBox.h"
#include "UserManager.h"

#include <QAction>
#include <QApplication>
#include <QByteArray>
#include <QColorDialog>
#include <QComboBox>
#include <QDialog>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSplitter>
#include <QTabWidget>
#include <QTextBrowser>
#include <QTextDocument>
#include <QTextEdit>
#include <QTextList>
#include <QToolButton>
#include <QVBoxLayout>
#include <QtConcurrent>

MailDockWidget::MailDockWidget(IAppContext *ctx, QObject *parent)
    : QObject(parent), m_ctx(ctx)
{
    m_mail = m_ctx->services()->get<IMailService>();
    loadFilterSettings();
    setupUi();
}

void MailDockWidget::setupUi()
{
    m_dock = new QDockWidget(m_ctx->translate("mail.dock_title"), m_ctx->mainWindow());
    m_dock->setObjectName("mailDockWidget");
    m_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea);
    QWidget* emptyTitle = new QWidget();
    emptyTitle->setFixedHeight(0);
    m_dock->setTitleBarWidget(emptyTitle); // Hide native title bar completely

    m_root = new QWidget(m_dock);
    m_root->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto *rootLayout = new QVBoxLayout(m_root);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(8);

    auto *tabs = new QTabWidget(m_root);
    tabs->setObjectName("mailTabs");
    tabs->addTab(buildInboxPage(), m_ctx->translate("mail.inbox"));
    tabs->addTab(buildComposePage(), m_ctx->translate("mail.compose"));
    rootLayout->addWidget(tabs);

    m_dock->setWidget(m_root);
    m_ctx->mainWindow()->addDockWidget(Qt::RightDockWidgetArea, m_dock);
    m_dock->hide();

    connect(m_refreshBtn, &QPushButton::clicked, this, &MailDockWidget::refreshInbox);
    connect(m_filterBtn, &QPushButton::clicked, this, &MailDockWidget::showFilterDialog);
    connect(m_sendBtn, &QPushButton::clicked, this, &MailDockWidget::sendCurrentMail);
}

QWidget *MailDockWidget::buildInboxPage()
{
    auto *page = new QWidget(m_root);
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto *toolbar = new QHBoxLayout();
    m_refreshBtn = new QPushButton(m_ctx->translate("mail.refresh"), page);
    m_filterBtn = new QPushButton(m_ctx->translate("mail.filter"), page);
    toolbar->addWidget(m_refreshBtn);
    toolbar->addWidget(m_filterBtn);
    toolbar->addStretch();
    layout->addLayout(toolbar);

    auto *splitter = new QSplitter(Qt::Horizontal, page);
    m_inboxList = new QListWidget(splitter);
    m_inboxList->setMinimumWidth(280);

    // Style inbox list — items are rendered via custom widgets, so keep base simple
    m_inboxList->setStyleSheet(
        "QListWidget { background-color: #1f2937; border: none; outline: none; }"
        "QListWidget::item { padding: 0px; margin: 3px 4px; border: 1px solid #374151; "
        "border-radius: 6px; background-color: #111827; }"
        "QListWidget::item:hover { background-color: #1e293b; border: 1px solid #6b7280; }"
        "QListWidget::item:selected { background-color: #1e3a5f; border: 1px solid #3b82f6; }"
    );

    m_preview = new QTextBrowser(splitter);
    m_preview->setOpenExternalLinks(true);
    m_preview->setStyleSheet(
        "QTextBrowser { background-color: #0f172a; color: #e2e8f0; border: none; "
        "padding: 20px; font-size: 14px; line-height: 1.6; }"
    );
    splitter->addWidget(m_inboxList);
    splitter->addWidget(m_preview);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    splitter->setSizes({320, 500});
    layout->addWidget(splitter, 1);

    connect(m_inboxList, &QListWidget::itemSelectionChanged, this, &MailDockWidget::onInboxSelectionChanged);
    return page;
}

QWidget *MailDockWidget::buildComposePage()
{
    auto *page = new QWidget(m_root);
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto *form = new QFormLayout();
    form->setLabelAlignment(Qt::AlignRight);
    m_toEdit = new QLineEdit(page);
    m_ccEdit = new QLineEdit(page);
    m_bccEdit = new QLineEdit(page);
    m_subjectEdit = new QLineEdit(page);
    form->addRow(m_ctx->translate("mail.to"), m_toEdit);
    form->addRow("Cc", m_ccEdit);
    form->addRow("Bcc", m_bccEdit);
    form->addRow(m_ctx->translate("mail.subject"), m_subjectEdit);
    layout->addLayout(form);

    layout->addWidget(buildFormatBar());

    m_bodyEdit = new QTextEdit(page);
    m_bodyEdit->setAcceptRichText(true);
    m_bodyEdit->setMinimumHeight(100);
    layout->addWidget(m_bodyEdit, 1);

    auto *attachmentRow = new QHBoxLayout();
    auto *attachBtn = new QPushButton(m_ctx->translate("mail.attach"), page);
    auto *removeAttachBtn = new QPushButton(m_ctx->translate("mail.remove_attachment"), page);
    m_attachmentList = new QListWidget(page);
    m_attachmentList->setMaximumHeight(82);
    attachmentRow->addWidget(attachBtn);
    attachmentRow->addWidget(removeAttachBtn);
    attachmentRow->addWidget(m_attachmentList, 1);
    layout->addLayout(attachmentRow);

    auto *actions = new QHBoxLayout();
    m_sendBtn = new QPushButton(m_ctx->translate("mail.send"), page);
    m_sendBtn->setObjectName("primary");
    actions->addStretch();
    actions->addWidget(m_sendBtn);
    layout->addLayout(actions);

    connect(attachBtn, &QPushButton::clicked, this, &MailDockWidget::addAttachments);
    connect(removeAttachBtn, &QPushButton::clicked, this, &MailDockWidget::removeSelectedAttachment);
    return page;
}

QWidget *MailDockWidget::buildFormatBar()
{
    auto *bar = new QWidget(m_root);
    auto *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    auto makeButton = [bar, layout](const QString &text, const QString &tip) {
        auto *button = new QToolButton(bar);
        button->setText(text);
        button->setToolTip(tip);
        button->setFixedSize(30, 28);
        layout->addWidget(button);
        return button;
    };

    auto *bold = makeButton("B", "Bold");
    bold->setCheckable(true);
    bold->setProperty("format", "bold");
    auto *italic = makeButton("I", "Italic");
    italic->setCheckable(true);
    italic->setProperty("format", "italic");
    auto *underline = makeButton("U", "Underline");
    underline->setCheckable(true);
    underline->setProperty("format", "underline");
    auto *color = makeButton("A", "Text color");
    color->setProperty("format", "color");
    auto *left = makeButton("<", "Align left");
    left->setProperty("format", "left");
    auto *center = makeButton("=", "Align center");
    center->setProperty("format", "center");
    auto *right = makeButton(">", "Align right");
    right->setProperty("format", "right");
    auto *bullet = makeButton("•", "Bullet list");
    bullet->setProperty("format", "bullet");
    auto *number = makeButton("1.", "Numbered list");
    number->setProperty("format", "number");
    auto *link = makeButton("@", "Insert link");
    link->setProperty("format", "link");

    m_fontSize = new QComboBox(bar);
    for (int size : {9, 10, 11, 12, 14, 16, 18, 22, 26}) {
        m_fontSize->addItem(QString::number(size), size);
    }
    m_fontSize->setCurrentText("12");
    layout->addWidget(m_fontSize);
    layout->addStretch();

    for (auto *button : bar->findChildren<QToolButton*>()) {
        connect(button, &QToolButton::clicked, this, &MailDockWidget::applyTextFormat);
    }
    connect(m_fontSize, &QComboBox::currentIndexChanged, this, [this]() {
        QTextCharFormat fmt;
        fmt.setFontPointSize(m_fontSize->currentData().toInt());
        m_bodyEdit->mergeCurrentCharFormat(fmt);
    });
    return bar;
}

void MailDockWidget::showCompose()
{
    m_dock->show();
    auto *tabs = m_root->findChild<QTabWidget*>("mailTabs");
    if (tabs) tabs->setCurrentIndex(1);
}

void MailDockWidget::showInbox()
{
    m_dock->show();
    auto *tabs = m_root->findChild<QTabWidget*>("mailTabs");
    if (tabs) tabs->setCurrentIndex(0);
    if (m_messages.isEmpty()) refreshInbox();
}

void MailDockWidget::setBusy(bool busy)
{
    if (m_refreshBtn) m_refreshBtn->setEnabled(!busy);
    if (m_sendBtn) m_sendBtn->setEnabled(!busy);
    if (busy) QApplication::setOverrideCursor(Qt::WaitCursor);
    else QApplication::restoreOverrideCursor();
}

MailMessage MailDockWidget::composeMessage() const
{
    MailMessage msg;
    msg.to = m_toEdit->text().trimmed();
    msg.cc = m_ccEdit->text().trimmed();
    msg.bcc = m_bccEdit->text().trimmed();
    msg.subject = m_subjectEdit->text().trimmed();
    msg.htmlBody = m_bodyEdit->toHtml();
    msg.attachmentPaths = m_attachments;
    return msg;
}

void MailDockWidget::sendCurrentMail()
{
    if (!m_mail) return;
    const MailMessage msg = composeMessage();
    if (msg.to.isEmpty()) {
        ModernMessageBox::warning(m_dock, m_ctx->translate("common.warning"), m_ctx->translate("mail.to_required"));
        return;
    }

    setBusy(true);
    auto *watcher = new QFutureWatcher<QPair<bool, QString>>(this);
    connect(watcher, &QFutureWatcher<QPair<bool, QString>>::finished, this, [this, watcher]() {
        const auto result = watcher->result();
        setBusy(false);
        watcher->deleteLater();
        if (result.first) {
            ModernMessageBox::information(m_dock, m_ctx->translate("common.success"), m_ctx->translate("mail.sent"));
            m_toEdit->clear();
            m_ccEdit->clear();
            m_bccEdit->clear();
            m_subjectEdit->clear();
            m_bodyEdit->clear();
            m_attachments.clear();
            populateAttachments();
        } else {
            ModernMessageBox::warning(m_dock, m_ctx->translate("common.error"), result.second);
        }
    });
    watcher->setFuture(QtConcurrent::run([this, msg]() {
        QString error;
        const bool ok = m_mail->sendMail(msg, error);
        return qMakePair(ok, error);
    }));
}

// Decode quoted-printable encoding (handles UTF-8 multi-byte and soft line breaks)
static QString decodeQuotedPrintable(const QString &input)
{
    QByteArray bytes;
    for (int i = 0; i < input.length(); ++i) {
        // Handle soft line breaks: = followed by \r\n or \n
        if (input[i] == '=' && i + 1 < input.length()) {
            if ((input[i + 1] == '\r' && i + 2 < input.length() && input[i + 2] == '\n') ||
                input[i + 1] == '\n') {
                // Skip the soft line break
                i += (input[i + 1] == '\r') ? 2 : 1;
                continue;
            }
            
            // Handle hex-encoded characters
            if (i + 2 < input.length()) {
                const QString hexStr = input.mid(i + 1, 2);
                bool ok = false;
                const int byte = hexStr.toInt(&ok, 16);
                if (ok && byte >= 0 && byte <= 255) {
                    bytes.append(static_cast<char>(byte));
                    i += 2;
                    continue;
                }
            }
        }
        bytes.append(input[i].toLatin1());
    }
    return QString::fromUtf8(bytes);
}

static bool looksLikeHtml(const QString &body)
{
    static const QRegularExpression htmlTagRe(
        "<\\s*/?\\s*(html|body|div|p|br|pre|span|table|tr|td|th|blockquote|b|strong|i|em|a)\\b",
        QRegularExpression::CaseInsensitiveOption);
    return htmlTagRe.match(body).hasMatch();
}

// Extract a clean display name from email From header
// "John Doe <john@example.com>" → "John Doe"
// "john@example.com" → "john@example.com"
static QString extractSenderName(const QString &from)
{
    const QString trimmed = from.trimmed();
    const int angleBracket = trimmed.indexOf('<');
    if (angleBracket > 0) {
        QString name = trimmed.left(angleBracket).trimmed();
        // Remove surrounding quotes
        if (name.startsWith('"') && name.endsWith('"'))
            name = name.mid(1, name.length() - 2).trimmed();
        if (!name.isEmpty()) return name;
    }
    // Fallback: return the email part or full string
    static const QRegularExpression emailRe("<([^>]+)>");
    const auto match = emailRe.match(trimmed);
    return match.hasMatch() ? match.captured(1) : trimmed;
}

static QString normalizeMailText(QString body)
{
    body = decodeQuotedPrintable(body);

    if (looksLikeHtml(body)) {
        QTextDocument doc;
        doc.setHtml(body);
        body = doc.toPlainText();
    }

    body.replace("\r\n", "\n");
    body.replace('\r', '\n');

    body.replace(QRegularExpression("\\s+(Vào\\s+[^\\n]{0,180}(?:đã\\s+)?viết\\s*:)",
                                    QRegularExpression::CaseInsensitiveOption),
                 "\n\n\\1\n");
    body.replace(QRegularExpression("\\s+(On\\s+[^\\n]{0,180}wrote\\s*:)",
                                    QRegularExpression::CaseInsensitiveOption),
                 "\n\n\\1\n");
    body.replace(QRegularExpression("\\s+((?:>\\s*)+)"), "\n\\1");

    return body;
}

static QString withoutQuoteMarkers(QString line)
{
    line = line.trimmed();
    while (line.startsWith('>')) {
        line = line.mid(1).trimmed();
    }
    return line;
}

static bool isMimeArtifactLine(const QString &line)
{
    const QString trimmed = withoutQuoteMarkers(line);
    return trimmed.startsWith("Content-", Qt::CaseInsensitive) ||
           trimmed.startsWith("MIME-", Qt::CaseInsensitive) ||
           trimmed.startsWith("This is a multi-part message", Qt::CaseInsensitive) ||
           trimmed.startsWith("--") ||
           trimmed.contains(QRegularExpression("--[A-Za-z0-9_=-]{12,}\\s+Content-",
                                               QRegularExpression::CaseInsensitiveOption)) ||
           trimmed.contains(QRegularExpression("\\bContent-(Type|Transfer-Encoding|Disposition):",
                                               QRegularExpression::CaseInsensitiveOption)) ||
           trimmed == "=";
}

// Clean email body: remove MIME headers and quoted-printable
static QString cleanEmailBody(const QString &body)
{
    QString cleanBody = normalizeMailText(body);
    
    // Remove MIME boundaries and structure
    cleanBody.remove(QRegularExpression("^--[a-f0-9]{24,}.*?$", QRegularExpression::MultilineOption));
    cleanBody.remove(QRegularExpression("--[a-f0-9]{24,}--", QRegularExpression::MultilineOption));
    
    // Remove IMAP response artifacts
    cleanBody.remove(QRegularExpression("^\\)\\s*$", QRegularExpression::MultilineOption));
    cleanBody.remove(QRegularExpression("BODY\\[HEADER\\.FIELDS[^\\)]*\\]", QRegularExpression::CaseInsensitiveOption));
    cleanBody.remove(QRegularExpression("\\{\\d+\\}", QRegularExpression::CaseInsensitiveOption));
    
    // Filter out header lines and metadata
    QStringList resultLines;
    for (const QString &line : cleanBody.split('\n')) {
        const QString trimmed = line.trimmed();
        
        if (isMimeArtifactLine(trimmed)) {
            if (!resultLines.isEmpty()) {
                break;
            }
            continue;
        }

        if (trimmed.startsWith("Date:", Qt::CaseInsensitive) ||
            trimmed.startsWith("Subject:", Qt::CaseInsensitive) ||
            trimmed.startsWith("To:", Qt::CaseInsensitive) ||
            trimmed.startsWith("From:", Qt::CaseInsensitive) ||
            trimmed.startsWith("Cc:", Qt::CaseInsensitive) ||
            trimmed.startsWith("Bcc:", Qt::CaseInsensitive) ||
            trimmed.isEmpty()) {
            continue;
        }
        resultLines.append(line);
    }
    
    return resultLines.join('\n').trimmed();
}

static QString renderMailBodyHtml(const QString &body)
{
    QString html;
    bool inQuote = false;
    bool quoteAfterReplyHeader = false;

    auto closeQuote = [&]() {
        if (inQuote) {
            html += "</div>";
            inQuote = false;
        }
    };

    for (QString line : body.split('\n')) {
        line = line.trimmed();
        if (line.isEmpty()) {
            closeQuote();
            continue;
        }

        int quoteDepth = 0;
        while (line.startsWith('>')) {
            ++quoteDepth;
            line = line.mid(1).trimmed();
        }
        if (line.isEmpty()) continue;

        const bool replyHeader =
            line.startsWith("Vào ", Qt::CaseInsensitive) ||
            line.startsWith("On ", Qt::CaseInsensitive);

        if (replyHeader) {
            closeQuote();
            html += QString("<div class='reply-header'>%1</div>").arg(line.toHtmlEscaped());
            quoteAfterReplyHeader = true;
            continue;
        }

        if (quoteAfterReplyHeader && quoteDepth == 0) {
            quoteDepth = 1;
        }

        if (quoteDepth > 0) {
            if (!inQuote) {
                html += "<div class='quoted-mail'>";
                inQuote = true;
            }
            const bool quoteMeta =
                line.startsWith("From:", Qt::CaseInsensitive) ||
                line.startsWith("To:", Qt::CaseInsensitive) ||
                line.startsWith("Date:", Qt::CaseInsensitive) ||
                line.startsWith("Subject:", Qt::CaseInsensitive);
            html += QString("<div class='%1'>%2</div>")
                        .arg(quoteMeta ? "quote-meta" : "quote-line",
                             line.toHtmlEscaped());
            continue;
        }

        closeQuote();
        html += QString("<p>%1</p>").arg(line.toHtmlEscaped());
    }

    closeQuote();
    return html.isEmpty() ? QString("<p></p>") : html;
}

void MailDockWidget::refreshInbox()
{
    if (!m_mail) return;
    setBusy(true);

    auto *progress = new CustomProgressDialog(m_ctx->mainWindow());
    progress->setLabelText(m_ctx->translate("mail.loading_inbox"));
    progress->setRange(0, 0);
    connect(progress, &CustomProgressDialog::stopRequested, progress, &CustomProgressDialog::hide);
    progress->show();
    progress->centerOnWidget(m_dock);
    QApplication::processEvents();

    auto *watcher = new QFutureWatcher<QPair<QList<MailMessage>, QString>>(this);
    connect(watcher, &QFutureWatcher<QPair<QList<MailMessage>, QString>>::finished, this, [this, watcher, progress]() {
        progress->hide();
        progress->deleteLater();
        const auto result = watcher->result();
        setBusy(false);
        watcher->deleteLater();
        if (!result.second.isEmpty()) {
            ModernMessageBox::warning(m_dock, m_ctx->translate("common.error"), result.second);
            return;
        }

        const QList<MailMessage> fullMessages = result.first;

        m_messages.clear();
        m_displayedIndices.clear();

        for (int i = 0; i < fullMessages.size(); ++i) {
            const MailMessage &msg = fullMessages[i];
            const QString searchableText = (msg.from + "\n" + msg.subject).toLower();
            bool shouldExclude = false;

            for (const QString &keyword : m_filterKeywords) {
                const QString normalizedKeyword = keyword.trimmed().toLower();
                if (!normalizedKeyword.isEmpty() && searchableText.contains(normalizedKeyword)) {
                    shouldExclude = true;
                    break;
                }
            }

            if (!shouldExclude) {
                m_displayedIndices.append(m_messages.size());
                m_messages.append(msg);
            }
        }

        if (m_messages.isEmpty() && !fullMessages.isEmpty()) {
            m_messages = fullMessages;
        }

        // Populate UI list with rich item widgets (Subject + Sender + Date)
        m_inboxList->clear();
        for (int i = 0; i < m_messages.size(); ++i) {
            const MailMessage &msg = m_messages[i];

            // Build a custom widget for each inbox item
            auto *itemWidget = new QWidget();
            auto *itemLayout = new QVBoxLayout(itemWidget);
            itemLayout->setContentsMargins(12, 8, 12, 8);
            itemLayout->setSpacing(3);

            // Row 1: Subject (with unread dot indicator)
            const QString subjectText = msg.subject.isEmpty()
                ? m_ctx->translate("mail.no_subject") : msg.subject;
            const QString unreadDot = msg.isRead ? "" : "● ";
            auto *subjectLabel = new QLabel(unreadDot + subjectText, itemWidget);
            subjectLabel->setStyleSheet(
                msg.isRead
                    ? "color: #e2e8f0; font-size: 13px;"
                    : "color: #f1f5f9; font-size: 13px; font-weight: bold;"
            );
            subjectLabel->setWordWrap(false);
            subjectLabel->setTextFormat(Qt::PlainText);
            itemLayout->addWidget(subjectLabel);

            // Row 2: Sender name
            const QString senderName = extractSenderName(msg.from);
            auto *senderLabel = new QLabel(senderName, itemWidget);
            senderLabel->setStyleSheet("color: #94a3b8; font-size: 11px;");
            senderLabel->setTextFormat(Qt::PlainText);
            itemLayout->addWidget(senderLabel);

            // Row 3: Date
            const QString dateStr = msg.date.isValid()
                ? msg.date.toString("dd/MM/yyyy  hh:mm") : QString();
            if (!dateStr.isEmpty()) {
                auto *dateLabel = new QLabel(dateStr, itemWidget);
                dateLabel->setStyleSheet("color: #64748b; font-size: 10px;");
                itemLayout->addWidget(dateLabel);
            }

            itemWidget->setLayout(itemLayout);

            // Create list item and attach the custom widget
            auto *item = new QListWidgetItem();
            item->setData(Qt::UserRole, msg.uid);
            item->setData(Qt::UserRole + 1, i);
            item->setToolTip(msg.from);
            item->setSizeHint(itemWidget->sizeHint());
            m_inboxList->addItem(item);
            m_inboxList->setItemWidget(item, itemWidget);
        }
    });
    watcher->setFuture(QtConcurrent::run([this]() {
        QString error;
        const QList<MailMessage> list = m_mail->fetchInbox(30, error);
        return qMakePair(list, error);
    }));
}

void MailDockWidget::onInboxSelectionChanged()
{
    QListWidgetItem *currentItem = m_inboxList->currentItem();
    if (!currentItem) return;

    // Get index from stored data (more reliable)
    bool ok = false;
    int row = currentItem->data(Qt::UserRole + 1).toInt(&ok);

    if (!ok || row < 0 || row >= m_messages.size()) return;

    const MailMessage &msg = m_messages[row];

    // Build clean header — dark mode compatible to match app theme
    const QString header = QString(
        "<style>"
        "body { font-family: 'Segoe UI', Arial, sans-serif; color: #e2e8f0; font-size: 14px; line-height: 1.7; "
        "       background-color: #0f172a; }"
        "h3 { font-size: 18px; font-weight: 600; margin: 0 0 14px 0; color: #f1f5f9; }"
        ".mail-header { background-color: #1e293b; padding: 18px 20px; border-radius: 10px; "
        "               margin-bottom: 20px; border: 1px solid #334155; }"
        ".mail-header-row { margin: 6px 0; font-size: 13px; }"
        ".mail-header-label { font-weight: 600; color: #94a3b8; display: inline-block; "
        "                     min-width: 55px; }"
        ".mail-header-value { color: #cbd5e1; }"
        ".mail-body { margin-top: 20px; word-wrap: break-word; color: #e2e8f0; }"
        ".mail-body p { margin: 8px 0; }"
        ".reply-header { margin: 20px 0 8px 0; padding: 8px 12px; color: #94a3b8; "
        "                font-weight: 600; font-size: 12px; background: #1e293b; "
        "                border-radius: 6px; }"
        ".quoted-mail { margin: 4px 0 0 0; padding: 10px 0 10px 14px; "
        "               border-left: 3px solid #3b82f6; color: #94a3b8; }"
        ".quote-line { margin: 3px 0; }"
        ".quote-meta { margin: 3px 0; color: #60a5fa; font-weight: 600; }"
        "a { color: #60a5fa; text-decoration: none; }"
        "a:hover { text-decoration: underline; color: #93c5fd; }"
        "table { border-collapse: collapse; width: 100%%; margin: 12px 0; }"
        "td, th { padding: 8px 12px; border: 1px solid #334155; color: #e2e8f0; }"
        "th { background-color: #1e293b; color: #f1f5f9; }"
        "pre { background-color: #1e293b; color: #e2e8f0; padding: 14px; "
        "      border-radius: 6px; overflow-x: auto; border: 1px solid #334155; }"
        "code { background-color: #1e293b; padding: 2px 6px; border-radius: 3px; "
        "       font-size: 13px; color: #f472b6; }"
        "img { max-width: 100%%; height: auto; border-radius: 6px; }"
        "hr { border: none; border-top: 1px solid #334155; margin: 16px 0; }"
        "</style>"
    );

    const QString mailHeader = QString(
        "<div class='mail-header'>"
        "<h3>%1</h3>"
        "<div class='mail-header-row'>"
        "  <span class='mail-header-label'>From:</span> "
        "  <span class='mail-header-value'>%2</span>"
        "</div>"
        "<div class='mail-header-row'>"
        "  <span class='mail-header-label'>To:</span> "
        "  <span class='mail-header-value'>%3</span>"
        "</div>"
        "<div class='mail-header-row'>"
        "  <span class='mail-header-label'>Date:</span> "
        "  <span class='mail-header-value'>%4</span>"
        "</div>"
        "</div>"
    ).arg(msg.subject.toHtmlEscaped(),
          msg.from.toHtmlEscaped(),
          msg.to.toHtmlEscaped(),
          msg.date.isValid() ? msg.date.toString("ddd, dd MMM yyyy  hh:mm") : QString());

    QString bodyHtml = msg.htmlBody.trimmed();
    if (bodyHtml.isEmpty()) {
        bodyHtml = "<p style='color:#64748b;font-style:italic;'>(Không có nội dung)</p>";
    } else if (!looksLikeHtml(bodyHtml)) {
        bodyHtml = renderMailBodyHtml(normalizeMailText(bodyHtml));
    }

    const QString html = header + mailHeader + "<div class='mail-body'>" + bodyHtml + "</div>";
    m_preview->setHtml(html);
}

void MailDockWidget::addAttachments()
{
    const QStringList files = QFileDialog::getOpenFileNames(m_dock, m_ctx->translate("file.select_file"));
    for (const QString &file : files) {
        if (!m_attachments.contains(file)) m_attachments << file;
    }
    populateAttachments();
}

void MailDockWidget::removeSelectedAttachment()
{
    const int row = m_attachmentList->currentRow();
    if (row >= 0 && row < m_attachments.size()) {
        m_attachments.removeAt(row);
        populateAttachments();
    }
}

void MailDockWidget::populateAttachments()
{
    m_attachmentList->clear();
    for (const QString &path : m_attachments) {
        QFileInfo info(path);
        m_attachmentList->addItem(info.fileName());
    }
}

void MailDockWidget::applyTextFormat()
{
    auto *button = qobject_cast<QToolButton*>(sender());
    if (!button || !m_bodyEdit) return;
    const QString format = button->property("format").toString();
    QTextCursor cursor = m_bodyEdit->textCursor();

    if (format == "bold" || format == "italic" || format == "underline") {
        QTextCharFormat fmt;
        if (format == "bold") fmt.setFontWeight(button->isChecked() ? QFont::Bold : QFont::Normal);
        if (format == "italic") fmt.setFontItalic(button->isChecked());
        if (format == "underline") fmt.setFontUnderline(button->isChecked());
        m_bodyEdit->mergeCurrentCharFormat(fmt);
    } else if (format == "color") {
        setTextColor();
    } else if (format == "left") {
        m_bodyEdit->setAlignment(Qt::AlignLeft);
    } else if (format == "center") {
        m_bodyEdit->setAlignment(Qt::AlignCenter);
    } else if (format == "right") {
        m_bodyEdit->setAlignment(Qt::AlignRight);
    } else if (format == "bullet") {
        cursor.createList(QTextListFormat::ListDisc);
    } else if (format == "number") {
        cursor.createList(QTextListFormat::ListDecimal);
    } else if (format == "link") {
        insertLink();
    }
}

void MailDockWidget::setTextColor()
{
    const QColor color = QColorDialog::getColor(Qt::black, m_dock, m_ctx->translate("mail.text_color"));
    if (!color.isValid()) return;
    QTextCharFormat fmt;
    fmt.setForeground(color);
    m_bodyEdit->mergeCurrentCharFormat(fmt);
}

void MailDockWidget::insertLink()
{
    bool ok = false;
    const QString url = QInputDialog::getText(m_dock, m_ctx->translate("mail.insert_link"),
                                              "URL:", QLineEdit::Normal, "https://", &ok);
    if (!ok || url.trimmed().isEmpty()) return;
    QTextCursor cursor = m_bodyEdit->textCursor();
    const QString text = cursor.selectedText().isEmpty() ? url : cursor.selectedText();
    cursor.insertHtml(QString("<a href=\"%1\">%2</a>").arg(url.toHtmlEscaped(), text.toHtmlEscaped()));
}

void MailDockWidget::showSettingsDialog()
{
    QDialog dlg(m_dock);
    dlg.setWindowTitle(m_ctx->translate("mail.settings"));
    dlg.setMinimumWidth(460);
    auto *layout = new QVBoxLayout(&dlg);
    auto *form = new QFormLayout();

    auto *email = new QLineEdit(m_mail ? m_mail->senderEmail() : QString(), &dlg);
    auto *password = new QLineEdit(&dlg);
    password->setEchoMode(QLineEdit::Password);
    auto *displayName = new QLineEdit(UserManager::instance()->currentUsername(), &dlg);
    auto *signature = new QTextEdit(&dlg);
    signature->setAcceptRichText(true);
    signature->setMinimumHeight(110);

    const QString username = UserManager::instance()->currentUsername();
    password->setText(UserManager::instance()->getUserPref(username, "mail_password"));
    displayName->setText(UserManager::instance()->getUserPref(username, "mail_display_name", username));
    signature->setHtml(UserManager::instance()->getUserPref(username, "mail_signature",
                                                            QString("<b>%1</b><br>3D-Reconstruction").arg(username)));

    form->addRow(m_ctx->translate("mail.account"), email);
    form->addRow(m_ctx->translate("mail.password"), password);
    form->addRow(m_ctx->translate("mail.display_name"), displayName);
    form->addRow(m_ctx->translate("mail.signature"), signature);
    layout->addLayout(form);

    auto *buttons = new QHBoxLayout();
    auto *testBtn = new QPushButton(m_ctx->translate("mail.test"), &dlg);
    auto *cancelBtn = new QPushButton(m_ctx->translate("common.cancel"), &dlg);
    auto *saveBtn = new QPushButton(m_ctx->translate("common.save"), &dlg);
    saveBtn->setObjectName("primary");
    buttons->addWidget(testBtn);
    buttons->addStretch();
    buttons->addWidget(cancelBtn);
    buttons->addWidget(saveBtn);
    layout->addLayout(buttons);

    connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    connect(saveBtn, &QPushButton::clicked, &dlg, [&]() {
        if (m_mail) {
            m_mail->setCredentials(email->text().trimmed(), password->text(), displayName->text().trimmed());
            UserManager::instance()->setUserPref(username, "mail_signature", signature->toHtml());
        }
        dlg.accept();
    });
    connect(testBtn, &QPushButton::clicked, &dlg, [&]() {
        if (!m_mail) return;
        m_mail->setCredentials(email->text().trimmed(), password->text(), displayName->text().trimmed());
        QString error;
        if (m_mail->testConnection(error)) {
            ModernMessageBox::information(&dlg, m_ctx->translate("common.success"), m_ctx->translate("mail.test_ok"));
        } else {
            ModernMessageBox::warning(&dlg, m_ctx->translate("common.error"), error);
        }
    });

    dlg.exec();
}

void MailDockWidget::loadFilterSettings()
{
    m_filterKeywords.clear();

    auto *um = UserManager::instance();
    if (um) {
        QString username = um->currentUsername();
        QString customFilters = um->getUserPref(username, "mail_filter_keywords", "");
        if (!customFilters.isEmpty()) {
            QStringList custom = customFilters.split("|", Qt::SkipEmptyParts);
            for (const QString &kw : custom) {
                if (!m_filterKeywords.contains(kw, Qt::CaseInsensitive)) {
                    m_filterKeywords << kw;
                }
            }
        }
    }
}

void MailDockWidget::saveFilterSettings()
{
    auto *um = UserManager::instance();
    if (!um) return;

    QStringList customKeywords;
    for (const QString &kw : m_filterKeywords) {
        const QString normalizedKeyword = kw.trimmed();
        if (!normalizedKeyword.isEmpty() &&
            !customKeywords.contains(normalizedKeyword, Qt::CaseInsensitive)) {
            customKeywords << normalizedKeyword;
        }
    }

    QString username = um->currentUsername();
    um->setUserPref(username, "mail_filter_keywords", customKeywords.join("|"));
}

void MailDockWidget::showFilterDialog()
{
    QDialog dlg(m_dock);
    dlg.setWindowTitle(m_ctx->translate("mail.filter"));
    dlg.setMinimumWidth(500);
    dlg.setMinimumHeight(400);
    auto *layout = new QVBoxLayout(&dlg);

    // Instructions
    auto *instruction = new QLabel(m_ctx->translate("mail.filter_hint"), &dlg);
    instruction->setStyleSheet("color: #888; font-size: 11px;");
    layout->addWidget(instruction);

    // Filter list
    auto *listLabel = new QLabel(m_ctx->translate("mail.filter_list"), &dlg);
    auto *filterList = new QListWidget(&dlg);
    for (const QString &kw : m_filterKeywords) {
        filterList->addItem(kw);
    }
    layout->addWidget(listLabel);
    layout->addWidget(filterList, 1);

    // Add/Remove buttons
    auto *btnRow = new QHBoxLayout();
    auto *addBtn = new QPushButton(m_ctx->translate("mail.add_filter"), &dlg);
    auto *removeBtn = new QPushButton(m_ctx->translate("mail.remove_filter"), &dlg);
    btnRow->addWidget(addBtn);
    btnRow->addWidget(removeBtn);
    btnRow->addStretch();
    layout->addLayout(btnRow);

    // Connect add button
    connect(addBtn, &QPushButton::clicked, &dlg, [this, &dlg, filterList]() {
        bool ok = false;
        QString keyword = QInputDialog::getText(&dlg,
            m_ctx->translate("mail.add_filter"),
            m_ctx->translate("mail.filter_input_hint"),
            QLineEdit::Normal, "", &ok);
        if (ok && !keyword.trimmed().isEmpty()) {
            keyword = keyword.trimmed();
            if (!m_filterKeywords.contains(keyword, Qt::CaseInsensitive)) {
                m_filterKeywords << keyword;
                filterList->addItem(keyword);
            }
        }
    });

    // Connect remove button
    connect(removeBtn, &QPushButton::clicked, &dlg, [this, filterList]() {
        int row = filterList->currentRow();
        if (row >= 0 && row < m_filterKeywords.size()) {
            m_filterKeywords.removeAt(row);
            delete filterList->takeItem(row);
        }
    });

    // Dialog buttons
    auto *buttons = new QHBoxLayout();
    auto *cancelBtn = new QPushButton(m_ctx->translate("common.cancel"), &dlg);
    auto *applyBtn = new QPushButton(m_ctx->translate("common.save"), &dlg);
    applyBtn->setObjectName("primary");
    buttons->addStretch();
    buttons->addWidget(cancelBtn);
    buttons->addWidget(applyBtn);
    layout->addLayout(buttons);

    connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    connect(applyBtn, &QPushButton::clicked, &dlg, [this, &dlg]() {
        saveFilterSettings();
        refreshInbox();
        dlg.accept();
    });

    dlg.exec();
}

void MailDockWidget::retranslate()
{
    if (m_dock) m_dock->setWindowTitle(m_ctx->translate("mail.dock_title"));
}
