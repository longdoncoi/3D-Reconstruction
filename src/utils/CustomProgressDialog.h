#ifndef CUSTOMPROGRESSDIALOG_H
#define CUSTOMPROGRESSDIALOG_H

#include "Global.h"
#include "IProgressDialog.h"

class ILanguageService;
class QLabel;
class QProgressBar;
class QPushButton;
class QWidget;

/**
 * @brief Concrete shell-owned progress dialog.
 *
 * Implements the plugin-facing IProgressDialog contract over a top-level
 * QWidget. It deliberately no longer derives from QDialog: a QObject-based
 * interface plus QDialog would introduce duplicate QObject bases, and the
 * dialog is only used modelessly via show()/hide() (never exec()).
 */
class APP_EXPORT CustomProgressDialog : public IProgressDialog {
    Q_OBJECT
public:
    explicit CustomProgressDialog(ILanguageService *lang, QWidget *parent = nullptr);
    ~CustomProgressDialog() override;

    void setRange(int minimum, int maximum) override;
    void setValue(int value) override;
    void setLabelText(const QString &text) override;
    void reset() override;
    void centerOnWidget(QWidget* target = nullptr) override;
    void show() override;
    void hide() override;

private slots:
    void onStopClicked();

private:
    QWidget *m_dialog = nullptr;
    QWidget *m_parent = nullptr;
    QLabel *lblTitle = nullptr;
    QProgressBar *progressBar = nullptr;
    QPushButton *btnStop = nullptr;
    ILanguageService *m_lang = nullptr;
};

#endif // CUSTOMPROGRESSDIALOG_H
