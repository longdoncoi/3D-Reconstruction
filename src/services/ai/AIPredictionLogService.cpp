#include "AIPredictionLogService.h"

#include "IAppPaths.h"

#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <exception>

namespace {

/// The production writer. `cv::imwrite` does not return false on every failure:
/// it THROWS `cv::Exception` when it cannot encode or open the destination, so
/// the call site below needs a barrier, not just a boolean check.
bool writeImage(const QString &path, const cv::Mat &image)
{
    return cv::imwrite(path.toStdString(), image);
}

} // namespace

AIPredictionLogService::AIPredictionLogService(IAppPaths *paths, QObject *parent)
    : QObject(parent)
    , m_paths(paths)
    , m_writer(&writeImage)
{
    // One worker: the writes are small and sequential; more threads would
    // only interleave the file names.
    m_pool.setMaxThreadCount(1);
    m_pool.setExpiryTimeout(30000);
}

AIPredictionLogService::~AIPredictionLogService()
{
    // The pool's destructor waits for the pending write. It is the last
    // thing that touches the service's state, so it happens here and not
    // somewhere in the middle of the teardown.
    //
    // Unbounded on purpose: the runnable below captures ONLY copies (path,
    // Mat, writer) — never `this` — so there is no lifetime hazard, and one
    // image encode is the worst case. A prediction log is a diagnostic side
    // effect; it must never outlive the service that scheduled it.
    m_pool.waitForDone(); // teardown-unbounded: the runnable captures copies only; one image encode is the worst case
}

void AIPredictionLogService::logPrediction(const QString &mode, const QString &sourceImagePath,
                                           const cv::Mat &resultImage)
{
    if (!m_paths || resultImage.empty()) return;

    const QString currentDate = QDateTime::currentDateTime().toString("yyyy-MM-dd");
    const QString predictDir = m_paths->predictDir(mode) + "/" + currentDate;
    if (!QDir().mkpath(predictDir)) {
        qWarning() << "[AIPredictionLog] could not create" << predictDir;
        return;
    }

    const QString logPath = predictDir + "/" + QFileInfo(sourceImagePath).baseName() + "_" +
                            QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";

    // The Mat is reference counted: the copy keeps the pixels alive for as
    // long as the worker needs them, whatever the caller does next.
    const cv::Mat resultClone = resultImage.clone();
    const Writer writer = m_writer;
    m_pool.start([logPath, resultClone, writer]() {
        // ── The barrier of a POOL thread ────────────────────────────────
        // A QRunnable runs inside QThread::run(): an exception that leaves it
        // reaches std::terminate() and aborts the whole application. cv::imwrite
        // throws on a failed encode or an unwritable path, so a prediction log
        // (a diagnostic side effect) could kill the process it is describing.
        // Logging must never break the inference that produced it — the
        // interface says so.
        try {
            if (!writer(logPath, resultClone))
                qWarning() << "[AIPredictionLog] could not write" << logPath;
        } catch (const std::exception &e) {
            qWarning() << "[AIPredictionLog] writing" << logPath << "failed:" << e.what();
        } catch (...) {
            qWarning() << "[AIPredictionLog] writing" << logPath
                       << "failed with an unknown error";
        }
    });
}

void AIPredictionLogService::setWriterForTesting(Writer writer)
{
    m_writer = writer ? std::move(writer) : Writer(&writeImage);
}
