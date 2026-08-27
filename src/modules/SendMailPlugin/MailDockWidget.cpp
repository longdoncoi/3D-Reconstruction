#include "MailDockWidget.h"

#include "MailFilterDialog.h"
#include "MailInboxItemFactory.h"
#include "MailMessageFormatter.h"
#include "MailSettingsDialog.h"
#include "IAppContext.h"
#include "CustomProgressDialog.h"
#include "ModernMessageBox.h"
#include "UserManager.h"

#include <QAction>
#include <QApplication>
#include <QByteArray>
#include <QColorDialog>
#include <QComboBox>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressDialog>
#include <QPushButton>
#include <QSplitter>
#include <QTabWidget>
#include <QTextBrowser>
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

        m_inboxList->clear();
        for (int i = 0; i < m_messages.size(); ++i) {
            MailInboxItemFactory::addMessageItem(
                m_inboxList,
                m_messages[i],
                i,
                m_ctx->translate("mail.no_subject"));
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

    m_preview->setHtml(MailMessageFormatter::previewHtml(
        msg,
        QString::fromUtf8("Kh\303\264ng c\303\263 n\341\273\231i dung")));

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
    MailSettingsDialog::show(m_dock, m_ctx, m_mail);
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
    if (MailFilterDialog::edit(m_dock, m_ctx, m_filterKeywords)) {
        saveFilterSettings();
        refreshInbox();
    }
}

void MailDockWidget::retranslate()
{
    if (m_dock) m_dock->setWindowTitle(m_ctx->translate("mail.dock_title"));
}
