#include "VideoTrackerThread.h"
#include <QDebug>
#include <QElapsedTimer>
#include <QMetaType>

VideoTrackerThread::VideoTrackerThread(IAIService* aiService, const QString& videoPath, QObject* parent)
    : QThread(parent), m_aiService(aiService), m_videoPath(videoPath), m_stopRequested(false), m_paused(false) {
    // frameProcessed(const cv::Mat&) được phát cross-thread (tracker → main)
    // theo queued connection, nên cv::Mat phải được đăng ký làm metatype —
    // nếu thiếu, Qt drop signal kèm warning "Cannot queue arguments of type
    // 'cv::Mat'" và view tracking không bao giờ nhận frame.
    qRegisterMetaType<cv::Mat>("cv::Mat");
}

VideoTrackerThread::~VideoTrackerThread() {
    stopAndWait();
}

void VideoTrackerThread::pause() {
    QMutexLocker locker(&m_mutex);
    m_paused = true;
}

void VideoTrackerThread::resume() {
    QMutexLocker locker(&m_mutex);
    m_paused = false;
    m_pauseCondition.wakeAll();
}

void VideoTrackerThread::stop() {
    QMutexLocker locker(&m_mutex);
    m_stopRequested = true;
    m_paused = false;
    m_pauseCondition.wakeAll();
}

bool VideoTrackerThread::stopAndWait(int timeoutMs) {
    stop();
    if (!isRunning())
        return true;

    if (wait(static_cast<unsigned long>(timeoutMs)))
        return true;

    qWarning() << "[VideoTrackerThread] Worker did not stop within"
               << timeoutMs << "ms; terminating it.";
    terminate();
    wait(2000);
    return false;
}

void VideoTrackerThread::run() {
    // ── The exception barrier of a THREAD ENTRY POINT ─────────────────────
    // An exception that leaves run() reaches std::terminate() and ABORTS THE
    // WHOLE PROCESS (0xC0000409). The ONNX inference itself is contained by
    // the service (ai::runInferenceGuarded), but cv::VideoCapture, the frame
    // handling and the clone() can throw too (cv::Exception, std::bad_alloc).
    // The barrier makes the thread's own failure a REPORTED failure: the
    // shell learns about it through errorOccurred and the "tracking" state is
    // reset through finishedTracking, instead of the app dying.
    try {
        cv::VideoCapture cap(m_videoPath.toStdString());
        if (!cap.isOpened()) {
            emit errorOccurred("Could not open video file.");
            return;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int delay = (fps > 0) ? (1000 / fps) : 30;

        cv::Mat frame;
        QElapsedTimer timer;

        while (true) {
            {
                QMutexLocker locker(&m_mutex);
                if (m_stopRequested) {
                    break;
                }
                while (m_paused) {
                    m_pauseCondition.wait(&m_mutex);
                    if (m_stopRequested) {
                        break;
                    }
                }
            }

            if (m_stopRequested) break;

            timer.start();

            if (!cap.read(frame)) {
                // End of video
                break;
            }

            if (frame.empty()) continue;

            // Run tracking
            cv::Mat trackedFrame = m_aiService->runTracking(frame);
            if (trackedFrame.empty()) {
                // The service reports a FAILED inference as an empty image (the
                // exception is contained there, nothing is thrown here). Sending
                // it on would paint a blank frame and leave the UI convinced
                // that tracking is running: a silent no-op instead of an error.
                emit errorOccurred(QStringLiteral(
                    "Tracking failed: the inference returned no result."));
                break;
            }

            emit frameProcessed(trackedFrame.clone());

            // Sleep to match FPS
            int elapsed = timer.elapsed();
            int sleepTime = delay - elapsed;
            if (sleepTime > 0) {
                msleep(sleepTime);
            }
        }

        cap.release();
        emit finishedTracking();
    } catch (const std::exception &e) {
        // Contained: the application survives and the failure is reported.
        emit errorOccurred(QStringLiteral("Tracking failed: %1")
                               .arg(QString::fromUtf8(e.what())));
        emit finishedTracking();
    } catch (...) {
        emit errorOccurred(QStringLiteral("Tracking failed with an unknown error."));
        emit finishedTracking();
    }
}
