// The global exception barrier test is deliberately NOT a QTEST_MAIN test:
// QTEST_MAIN constructs a plain QApplication, and the subject under test is
// the QApplication subclass itself. This executable is its own `main()`:
//
//   * post a queued slot that throws,
//   * let the event loop dispatch it through GuardedApplication::notify(),
//   * quit afterwards.
//
// If the barrier is missing or disabled, the exception escapes notify(), Qt
// calls std::terminate() and the process aborts — CTest reports the crash. If
// the barrier holds, `exec()` returns 0 and the process exits 0. That is the
// whole assertion: surviving is the behaviour.

#include <QCoreApplication>
#include <QTimer>

#include <cstdio>
#include <stdexcept>

#include "GuardedApplication.h"

class ThrowingSlot : public QObject {
    Q_OBJECT

public slots:
    void boom() {
        throw std::runtime_error("exception from a queued slot");
    }
};

int main(int argc, char** argv) {
    GuardedApplication app(argc, argv);

    ThrowingSlot thrower;
    QTimer::singleShot(0, &thrower, &ThrowingSlot::boom);
    QTimer::singleShot(0, &app, &QCoreApplication::quit);

    const int rc = app.exec();
    std::printf("event loop survived a throwing slot (rc=%d)\n", rc);
    return rc == 0 ? 0 : 1;
}

#include "TestGuardedApplication.moc"
