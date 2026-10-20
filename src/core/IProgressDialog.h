#ifndef IPROGRESSDIALOG_H
#define IPROGRESSDIALOG_H

#include "Global.h"

#include <QObject>
#include <QString>

class QWidget;

/**
 * @brief Plugin-facing contract for the shell-owned progress dialog.
 *
 * The concrete dialog lives in AppCore (src/utils) and is created by
 * MainWindow. Plugins must only see this interface: the previous contract
 * exposed `CustomProgressDialog` from IShellService, which leaked a concrete
 * AppCore class into every plugin's ABI and forced plugins to include a
 * widget header just to connect one signal.
 *
 * Emission of stopRequested is owned by the implementation; callers connect
 * to it through this interface.
 */
class APP_EXPORT IProgressDialog : public QObject {
    Q_OBJECT

public:
    explicit IProgressDialog(QObject* parent = nullptr) : QObject(parent) {}
    ~IProgressDialog() override = default;

    virtual void setRange(int minimum, int maximum) = 0;
    virtual void setValue(int value) = 0;
    virtual void setLabelText(const QString &text) = 0;
    virtual void reset() = 0;
    virtual void centerOnWidget(QWidget* target = nullptr) = 0;
    virtual void show() = 0;
    virtual void hide() = 0;

signals:
    /** User pressed Stop; operations should cancel cooperatively. */
    void stopRequested();
};

#endif // IPROGRESSDIALOG_H
