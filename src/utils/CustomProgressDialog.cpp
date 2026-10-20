#include "CustomProgressDialog.h"
#include "ILanguageService.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPoint>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

CustomProgressDialog::CustomProgressDialog(ILanguageService *lang, QWidget *parent)
    : IProgressDialog(parent), m_parent(parent), m_lang(lang) {
    m_dialog = new QWidget();
    m_dialog->setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint | Qt::WindowSystemMenuHint);
    m_dialog->setAttribute(Qt::WA_TranslucentBackground);
    m_dialog->setWindowModality(Qt::ApplicationModal);
    m_dialog->setStyleSheet(
        "#progressContainer { background-color: #1a1a1f; border: 1px solid #3a3a4a; border-radius: 8px; }");
    m_dialog->setMinimumWidth(500);

    auto *baseLayout = new QVBoxLayout(m_dialog);
    baseLayout->setContentsMargins(10, 10, 10, 10);

    auto *container = new QWidget(m_dialog);
    container->setObjectName("progressContainer");
    baseLayout->addWidget(container);

    auto *mainLayout = new QVBoxLayout(container);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(15);

    // Dòng 1: Title và Nút Stop
    auto *topLayout = new QHBoxLayout();

    lblTitle = new QLabel(m_lang->translate("common.loading"), container);
    lblTitle->setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;");

    btnStop = new QPushButton(m_lang->translate("progress.stop"), container);
    btnStop->setCursor(Qt::PointingHandCursor);
    btnStop->setStyleSheet(
        "QPushButton { background:#d32f2f; color:#fff; border:none; border-radius:4px; padding:6px 16px; font-weight:bold; font-size: 13px; }"
        "QPushButton:hover { background:#b71c1c; }");
    connect(btnStop, &QPushButton::clicked, this, &CustomProgressDialog::onStopClicked);
    connect(m_lang, &ILanguageService::languageChanged, this, [this]() {
        btnStop->setText(m_lang->translate("progress.stop"));
    });

    topLayout->addWidget(lblTitle);
    topLayout->addStretch();
    topLayout->addWidget(btnStop);

    // Dòng 2: ProgressBar
    progressBar = new QProgressBar(container);
    progressBar->setFixedHeight(12);
    progressBar->setTextVisible(false);
    progressBar->setRange(0, 0);
    progressBar->setStyleSheet(
        "QProgressBar { background:#2a2a35; border:none; border-radius:6px; }"
        "QProgressBar::chunk { background-color: #10a37f; border-radius:6px; }");

    mainLayout->addLayout(topLayout);
    mainLayout->addWidget(progressBar);
}

CustomProgressDialog::~CustomProgressDialog() {
    delete m_dialog;
    m_dialog = nullptr;
}

void CustomProgressDialog::setRange(int minimum, int maximum) {
    progressBar->setRange(minimum, maximum);
}

void CustomProgressDialog::setValue(int value) {
    progressBar->setValue(value);
}

void CustomProgressDialog::setLabelText(const QString &text) {
    lblTitle->setText(text);
}

void CustomProgressDialog::reset() {
    progressBar->reset();
}

void CustomProgressDialog::centerOnWidget(QWidget* target) {
    if (!target)
        target = m_parent;
    if (target) {
        m_dialog->adjustSize();
        const QPoint center = target->mapToGlobal(target->rect().center());
        m_dialog->move(center.x() - m_dialog->width() / 2,
                       center.y() - m_dialog->height() / 2);
    }
}

void CustomProgressDialog::show() {
    m_dialog->show();
    m_dialog->raise();
}

void CustomProgressDialog::hide() {
    m_dialog->hide();
}

void CustomProgressDialog::onStopClicked() {
    emit stopRequested();
}
