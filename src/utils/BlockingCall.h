#ifndef BLOCKINGCALL_H
#define BLOCKINGCALL_H

#include <QFuture>
#include <QLoggingCategory>
#include <QObject>
#include <QPromise>
#include <QSemaphore>
#include <QThread>

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

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
 *
 * A task that outlives the objects it uses is a use-after-free waiting to
 * happen: the service it calls is destroyed when the shell tears down. Pass a
 * `TaskTracker` to `run()` and wait for it in the teardown path — the wait
 * is bounded, so a hung task cannot freeze the shutdown.
 *
 * ── The two guarantees this helper makes ──────────────────────────────────────
 *
 * 1. **The promise is finished EXPLICITLY.** `QPromise::~QPromise()` calls
 *    `cancelAndFinish()` on a promise that was never finished, so a helper
 *    that only calls `addResult()` delivers its result in the *cancelled*
 *    state — and only when the promise happens to be destroyed (i.e. when the
 *    worker thread object is torn down). A cancelled future is a landmine:
 *    the natural `if (watcher.isCanceled()) return;` guard silently drops the
 *    result. `run()` calls `finish()` itself, so the future is *completed* and
 *    the notification does not depend on the thread object's lifetime.
 *
 * 2. **An exception never escapes the worker thread.** ONNX Runtime, OpenCV
 *    and the socket layer all throw. An exception that leaves `QThread::run()`
 *    reaches `std::terminate()` and aborts the whole process — the app dies
 *    with 0xC0000409 instead of reporting a failed request. `run()` contains
 *    the exception, logs it and always releases the `TaskTracker` semaphore.
 *
 *    The exception is deliberately **not** stored on the promise: `QFuture::
 *    result()` re-throws an exception stored there, and that re-throw happens
 *    inside the *caller's* slot — which ends in `std::terminate()` again, one
 *    stack frame further from the cause. Instead the failure is delivered as
 *    a **default-constructed result**: `QFuture::result()` reads the result
 *    store unconditionally, so a future that finished WITHOUT a result makes
 *    the caller read out of bounds. A caller that must report a domain error
 *    catches it inside the callable and returns it in the result (the mail
 *    stack does exactly that: `errorMsg` is filled by the service).
 */
namespace blocking {

/**
 * @brief Keeps track of the background tasks a component started.
 *
 * The owner waits for the tasks in its teardown path, *before* it releases
 * the objects the tasks use. The wait is bounded: a task that hangs must not
 * freeze the shutdown.
 */
class TaskTracker {
public:
    TaskTracker() = default;
    TaskTracker(const TaskTracker &) = delete;
    TaskTracker &operator=(const TaskTracker &) = delete;

    /// Wait for every tracked task, at most @p timeoutMsPerTask each.
    /// Returns the number of tasks that did not finish in time.
    int waitForFinished(int timeoutMsPerTask)
    {
        int unfinished = 0;
        for (const auto &done : m_tasks) {
            if (!done->tryAcquire(1, timeoutMsPerTask)) ++unfinished;
        }
        m_tasks.clear();
        return unfinished;
    }

    /// The number of tasks that were started and not waited for yet.
    int taskCount() const { return static_cast<int>(m_tasks.size()); }

    /// Drop the tasks that have already completed; keep the running ones.
    ///
    /// An owner that starts a sequence of tasks on the same tracker (a model
    /// reload, a second mail request) calls this before it registers the next
    /// one, so the tracker does not grow by one semaphore per task for the
    /// whole session. A task that is still running stays tracked and is still
    /// joined by waitForFinished()/waitForAll().
    /// @return the number of tasks that were still running.
    int reapFinished()
    {
        int running = 0;
        std::vector<std::shared_ptr<QSemaphore>> kept;
        kept.reserve(m_tasks.size());
        for (auto &done : m_tasks) {
            if (done->tryAcquire(1)) continue; // completed: release() happened
            kept.push_back(std::move(done));
            ++running;
        }
        m_tasks.swap(kept);
        return running;
    }

    /// Wait for every tracked task with no timeout.
    ///
    /// This is deliberately UNBOUNDED and reserved for a teardown path whose
    /// task captures the owner (`this`) or a collaborator the owner is about
    /// to destroy. A bounded wait would return with the task still running and
    /// let it touch freed memory; the process is exiting anyway. Every other
    /// teardown path uses the bounded waitForFinished(ms).
    void waitForAll()
    {
        for (const auto &done : m_tasks) {
            done->acquire();
        }
        m_tasks.clear();
    }

    /// Register a running task. Called by run() — the tracker keeps the
    /// completion signal alive.
    void track(std::shared_ptr<QSemaphore> done) { m_tasks.push_back(std::move(done)); }

private:
    std::vector<std::shared_ptr<QSemaphore>> m_tasks;
};

/// Runs @p callable on a dedicated thread and returns its result.
/// When @p tracker is given, the task is registered with it, so the owner
/// can wait for it before it tears down what the task uses.
///
/// The returned future is **completed** (not cancelled) when the callable
/// returns. When the callable throws, the exception is contained and logged,
/// and the future completes with a default-constructed result.
template <typename Callable>
auto run(Callable &&callable, TaskTracker *tracker = nullptr) -> QFuture<std::invoke_result_t<Callable>>
{
    using Result = std::invoke_result_t<Callable>;

    // The failure path has to deliver a VALUE: QFuture::result() reads the
    // result store unconditionally, so a future that finished without a
    // result makes the caller read out of bounds.
    static_assert(std::is_void_v<Result> || std::is_default_constructible_v<Result>,
                  "blocking::run needs a default-constructible result type: the "
                  "failure path must be able to deliver a value to the caller.");

    auto promise = std::make_shared<QPromise<Result>>();
    auto future = promise->future();

    // The completion signal of the task: released by the worker thread when
    // the callable returns, whatever its result.
    auto done = std::make_shared<QSemaphore>();
    if (tracker) tracker->track(done);

    auto *thread = QThread::create(
        [promise, done, fn = std::forward<Callable>(callable)]() mutable {
            // ── The exception barrier ──────────────────────────────────────
            // An exception must not leave this lambda: it would escape
            // QThread::run() and reach std::terminate(), killing the whole
            // application instead of failing one request. The failure is
            // logged and delivered as a default-constructed result — the
            // "did it work?" value for the result types used here.
            try {
                if constexpr (std::is_void_v<Result>) {
                    fn();
                } else {
                    promise->addResult(fn());
                }
            } catch (const std::exception &e) {
                qWarning() << "blocking::run: the task failed:" << e.what();
                if constexpr (!std::is_void_v<Result>) {
                    promise->addResult(Result{});
                }
            } catch (...) {
                qWarning() << "blocking::run: the task failed with an unknown error.";
                if constexpr (!std::is_void_v<Result>) {
                    promise->addResult(Result{});
                }
            }
            // ── The explicit completion ─────────────────────────────────────
            // The promise is finished HERE, not by its destructor: a promise
            // that is destroyed unfinished is cancelled (QPromise's
            // destructor calls cancelAndFinish()), and a cancelled future is
            // indistinguishable from a failed one for the caller.
            promise->finish();
            // Always released — a throwing callable must not make the owner
            // wait for the full timeout of its teardown.
            done->release();
        });
    QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
    return future;
}

} // namespace blocking

#endif // BLOCKINGCALL_H
