#ifndef GUARDEDAPPLICATION_H
#define GUARDEDAPPLICATION_H

#include <QApplication>
#include <QDebug>

#include <exception>

/**
 * @brief QApplication with a process-wide exception barrier around event
 *        dispatch.
 *
 * An exception that leaves a Qt slot reaches std::terminate() and aborts the
 * whole application (0xC0000409): no destructor runs and the log loses the
 * cause. Qt's event loop is the single funnel every GUI entry point goes
 * through — widget events, timers, queued connections, `QMetaObject::
 * invokeMethod` — so containing the exception here turns a process abort into
 * a logged failure while the loop keeps running.
 *
 * This is the GUI-thread half of the barrier, not a replacement for the other
 * half: worker threads never pass through `notify()`, so `blocking::run`,
 * `ai::runInferenceGuarded` and every `QThread::run()` implementation must
 * keep their own barrier (see Docs/ARCHITECTURE_SRC.md, "Exception barriers").
 */
class GuardedApplication : public QApplication {
public:
    using QApplication::QApplication;

    bool notify(QObject* receiver, QEvent* event) override {
        try {
            return QApplication::notify(receiver, event);
        } catch (const std::exception& e) {
            qCritical().noquote()
                << "GuardedApplication: contained exception from event dispatch:"
                << e.what();
        } catch (...) {
            qCritical().noquote()
                << "GuardedApplication: contained a non-standard exception from event dispatch";
        }
        // The event was consumed. Returning false would make Qt forward it to
        // the parent widget, which runs the same broken code a second time.
        return true;
    }
};

#endif // GUARDEDAPPLICATION_H
