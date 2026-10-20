#ifndef VIDEOTRACKERTHREAD_H
#define VIDEOTRACKERTHREAD_H

#include <QThread>
#include <QString>
#include <QMutex>
#include <QWaitCondition>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "IAIService.h"

/**
 * Vòng lặp tracking video chạy trên thread riêng.
 *
 * CỐ Ý giữ QThread subclass (không dùng worker-object + moveToThread):
 * run() là vòng lặp blocking với pause/resume qua QMutex/QWaitCondition.
 * Nếu chuyển sang worker-object, pause()/stop() gửi qua queued connection
 * sẽ không bao giờ được xử lý trong lúc slot xử lý frame đang block event
 * loop của worker — trong khi gọi method trực tiếp có mutex bảo vệ ở đây
 * là mô hình đúng cho blocking loop (xem Qt docs "When to subclass QThread").
 *
 * frameProcessed(const cv::Mat&) là queued cross-thread → ctor đăng ký
 * metatype cv::Mat (qRegisterMetaType) để signal không bị drop.
 */
class VideoTrackerThread : public QThread {
    Q_OBJECT
public:
    explicit VideoTrackerThread(IAIService* aiService, const QString& videoPath, QObject* parent = nullptr);
    ~VideoTrackerThread() override;

    void pause();
    void resume();
    void stop();

    /**
     * Request stop and join with a bounded wait. Keeping the timeout explicit
     * prevents shutdown from blocking forever if a frame is stuck inside the
     * inference backend. A stuck thread is terminated only as a last resort
     * (same policy as ReconstructionService's worker teardown).
     */
    bool stopAndWait(int timeoutMs = 15000);

signals:
    void frameProcessed(const cv::Mat& frame);
    void finishedTracking();
    void errorOccurred(const QString& errorMsg);

protected:
    void run() override;

private:
    IAIService* m_aiService;
    QString m_videoPath;
    
    QMutex m_mutex;
    QWaitCondition m_pauseCondition;
    
    // Read in run() both under the mutex (pause loop) and outside it
    // (per-frame check), so it must be atomic to avoid a data race with
    // stop() being called from the GUI thread.
    std::atomic_bool m_stopRequested;
    bool m_paused;
};

#endif // VIDEOTRACKERTHREAD_H
