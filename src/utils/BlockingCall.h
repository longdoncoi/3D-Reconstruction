#ifndef BLOCKINGCALL_H
#define BLOCKINGCALL_H

#include <QFuture>
#include <QObject>
#include <QPromise>
#include <QThread>

#include <memory>
#include <type_traits>
#include <utility>

/**
 * @brief Runs a blocking call on its own thread and returns its result.
 *
 * `QtConcurrent::run` does **not** guarantee that the task leaves the calling
 * thread: when the pool has no free worker, Qt may execute the task inline
 * (QThreadPool::doStart falls back to an inline run).  For a blocking network
 * call — SMTP, IMAP, HTTP — that means the GUI thread freezes for the whole
 * socket timeout, which is exactly what the mail stack's
 * `mail::enforceBackgroundThread()` guards against.
 *
 * `blocking::run` always uses a dedicated thread, so the caller can rely on
 * "never on the GUI thread" and the guard stays a last line of defence rather
 * than the primary mechanism.
 *
 * Usage (the result is delivered to the caller's thread):
 *
 *     QFutureWatcher<Result> watcher;
 *     connect(&watcher, &QFutureWatcherBase::finished, this, [this, &watcher] {
 *         const Result result = watcher.result();
 *         ...
 *     });
 *     watcher.setFuture(blocking::run([service] {
 *         return service->doSomethingBlocking();
 *     }));
 */
namespace blocking {

template <typename Callable>
auto run(Callable &&callable) -> QFuture<std::invoke_result_t<Callable>>
{
    using Result = std::invoke_result_t<Callable>;

    auto promise = std::make_shared<QPromise<Result>>();
    auto future = promise->future();

    auto *thread = QThread::create(
        [promise, fn = std::forward<Callable>(callable)]() mutable {
            if constexpr (std::is_void_v<Result>) {
                fn();
                promise->finish();
            } else {
                promise->addResult(fn());
            }
        });
    QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
    return future;
}

} // namespace blocking

#endif // BLOCKINGCALL_H
