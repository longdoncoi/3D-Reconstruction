#include "ReconstructionService.h"
#include "ReconstructionWorker.h"
#include <QDir>
#include <opencv2/core.hpp>

ReconstructionService::ReconstructionService()
    : m_pipeline(std::make_unique<ReconstructionPipeline>())
{
}

ReconstructionService::~ReconstructionService()
{
    // Ngừng nhận tín hiệu từ worker: queued invocation nhắm vào `this` sẽ bị
    // QObject destructor gỡ khỏi event queue, nên không có callback sau hủy.
    if (m_worker)
        m_worker->disconnect(this);

    if (m_thread) {
        // Hủy cooperative tại checkpoint kế tiếp; quit() thread-safe từ mọi
        // thread (không cần event loop của service xử lý).
        m_cancelRequested = true;
        m_thread->quit();
        if (!m_thread->wait(15000)) {
            // Fallback cuối: chỉ khi pipeline mắc kẹt trong thư viện ngoài
            // (Poisson/PCL...) — app đang thoát nên rủi ro giới hạn.
            m_thread->terminate();
            // The argument-less wait() blocks for Qt's 30 s default (it does
            // not wait forever, but it is the longest wait in the whole
            // teardown path). terminate() has already been requested; waiting
            // longer than the termination itself cannot help.
            m_thread->wait(2000);
        }
    }

    // Thread đã dừng hẳn (hoặc chưa từng chạy) → xóa worker/thread an toàn.
    // Nếu finishRun() đã chạy trước destructor, hai con trỏ đã là nullptr.
    delete m_worker;
    delete m_thread;
    m_worker = nullptr;
    m_thread = nullptr;
}

void ReconstructionService::setImages(const QStringList& paths) {
    std::vector<QString> stdPaths;
    stdPaths.reserve(static_cast<size_t>(paths.size()));
    for (const QString& p : paths)
        stdPaths.push_back(QDir::toNativeSeparators(p));

    // The pipeline lock is held by the worker for the WHOLE reconstruction.
    // Blocking on it here would freeze the GUI for the entire run — the very
    // defect the published/pipeline split exists to prevent. The call is
    // REFUSED instead: the same intent as the old `if (m_running) return;`,
    // but without waiting for the run to end first.
    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        qWarning() << "ReconstructionService: the pipeline is busy — the image list was not changed.";
        return;
    }
    // Re-check UNDER the lock: the worker may have started between the check
    // and the acquisition (check-then-act). m_running is only a hint — the
    // lock is the authority.
    if (m_running) {
        qWarning() << "ReconstructionService: a reconstruction is running — the image list was not changed.";
        return;
    }
    m_pipeline->setImages(stdPaths);
    // Publish under the published lock only: the pipeline lock (held above) is
    // released on return, and the UI must never contend with the worker.
    std::lock_guard pub(m_publishedMutex);
    m_snapshot.reset();
    m_publishedPointCount = 0;
    m_imageList.clear();
    for (const auto &path : m_pipeline->getImages())
        m_imageList.append(QString::fromStdString(path.toStdString()));
}

bool ReconstructionService::loadCameraParams(const QString& filePath) {
    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        qWarning() << "ReconstructionService: the pipeline is busy — the camera parameters were not loaded.";
        return false;
    }
    if (m_running) return false;
    const bool ok = m_pipeline->loadCameraParams(filePath);
    if (ok) {
        std::lock_guard pub(m_publishedMutex);
        m_snapshot.reset();
        m_publishedPointCount = 0;
    }
    return ok;
}

void ReconstructionService::configure(int densityLevel, int maxFeatures, bool autoClean) {
    // Non-blocking, like setImages(): the GUI must not wait for a running
    // reconstruction to release the pipeline lock.
    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        qWarning() << "ReconstructionService: the pipeline is busy — the configuration was not applied.";
        return;
    }
    if (m_running) return;
    
    ReconstructionConfig cfg = m_pipeline->config();
    cfg.sift.nfeatures = maxFeatures;
    
    // Map density Level (0: Thấp, 1: TB, 2: Cao, 3: Rất cao) to leafSize
    switch (densityLevel) {
        case 0: cfg.filter.voxelLeafSizeTrack = 0.005f; break; // Thấp
        case 1: cfg.filter.voxelLeafSizeTrack = 0.001f; break; // TB
        case 2: cfg.filter.voxelLeafSizeTrack = 0.0005f; break;// Cao
        case 3: cfg.filter.voxelLeafSizeTrack = 0.0001f; break;// Rất cao
        default: cfg.filter.voxelLeafSizeTrack = 0.001f; break;
    }
    
    if (!autoClean) {
        cfg.filter.sorMeanKTrack = 0; // Disable SOR
        cfg.filter.rorMinNeighborsTrack = 0; // Disable ROR
    } else {
        cfg.filter.sorMeanKTrack = 15;
        cfg.filter.rorMinNeighborsTrack = 2;
    }
    
    m_pipeline->setConfig(cfg);
}

bool ReconstructionService::reconstruct() {
    if (m_running.exchange(true)) return false; // Already running
    m_cancelRequested = false; // cờ hủy thuộc đúng một lần chạy
    {
        // The previous result is no longer current; drop the cached snapshot
        // before the worker starts mutating the pipeline. The published lock
        // is enough — the pipeline lock belongs to the worker from here on.
        std::lock_guard pub(m_publishedMutex);
        m_snapshot.reset();
        m_publishedPointCount = 0;
    }

    // Worker được move vào QThread nội bộ (service layer sở hữu thread,
    // không bao giờ thuộc về plugin). Worker emit finished(bool) từ
    // thread chạy pipeline → handler dưới đây chạy queued trên thread
    // của service (receiver `this`). Thread chỉ thoát exec() khi worker
    // báo xong (quit), và self-delete sau khi đã kết thúc.
    m_thread = new QThread(); // ownership tường minh (finishRun/dtor xóa)
    auto* worker = new ReconstructionWorker(this); // không parent → mới move được
    m_worker = worker;
    worker->moveToThread(m_thread);

    // Worker tự quit event loop của chính nó ngay khi run() trả về
    // (DirectConnection: quit() thread-safe, không phụ thuộc event loop của
    // service). Teardown chạy queued trên thread của service.
    connect(worker, &ReconstructionWorker::finished, m_thread,
            &QThread::quit, Qt::DirectConnection);
    connect(worker, &ReconstructionWorker::finished, this,
            [this](bool ok) { finishRun(ok); }, Qt::QueuedConnection);

    QMetaObject::invokeMethod(worker, "run", Qt::QueuedConnection);
    m_thread->start();
    return true;
}

void ReconstructionService::finishRun(bool ok) {
    // Thread đã được quit() ngay trước tín hiệu; join để chắc chắn nó dừng
    // hẳn rồi mới xóa worker (worker có affinity với thread đó).
    //
    // The wait is BOUNDED: this runs on the GUI thread, and a pipeline stuck
    // in an external library (PCL/FLANN) would otherwise freeze the window
    // forever. The fallback is the same one the destructor uses.
    //
    // The terminate() below is reachable only in the pathological case: this
    // slot runs on the service's thread AFTER the worker emitted finished(),
    // i.e. after runPipeline() returned and released m_mutex — a normal run
    // never gets here. It stays bounded because the pipeline lock may have
    // been abandoned mid-run by an external library, and waiting on it is
    // exactly what would freeze the GUI.
    if (m_thread) {
        if (!m_thread->wait(15000)) {
            qWarning() << "Reconstruction: the worker did not stop in time; terminating it.";
            m_thread->terminate();
            m_thread->wait(2000);
        }
        delete m_thread;
        m_thread = nullptr;
    }
    delete m_worker;
    m_worker = nullptr;
    m_running = false;

    QString detail;
    {
        std::shared_lock lock(m_mutex);
        detail = m_lastError;
    }
    emit reconstructFinished(ok, ok ? QString() : detail);
}

bool ReconstructionService::runPipeline() {
    bool ok = false;
    {
        std::unique_lock lock(m_mutex);
        // The pipeline calls into OpenCV and PCL, and both throw. An
        // exception escaping the worker thread would reach std::terminate
        // and take the whole application down — instead of the
        // reconstructFinished(false, message) the interface promises.
        try {
            ok = m_pipeline->reconstruct(
                [this] { return m_cancelRequested.load(std::memory_order_relaxed); });
            m_lastError = ok ? QString() : m_pipeline->errorString();
        } catch (const cv::Exception &e) {
            ok = false;
            m_lastError = QStringLiteral("OpenCV error during reconstruction: %1")
                              .arg(QString::fromUtf8(e.what()));
            qWarning() << m_lastError;
        } catch (const std::exception &e) {
            ok = false;
            m_lastError = QStringLiteral("Reconstruction failed: %1")
                              .arg(QString::fromUtf8(e.what()));
            qWarning() << m_lastError;
        } catch (...) {
            ok = false;
            m_lastError = QStringLiteral("Reconstruction failed with an unknown error.");
            qWarning() << m_lastError;
        }
        // Publish the result for the UI — HERE, on the worker thread, while
        // the pipeline lock is held. The UI then reads the published copy and
        // never has to touch the pipeline (which is locked for the whole run).
        // Building the snapshot here also keeps the copy off the GUI thread.
        //
        // It is published on EVERY run, not only on a successful one: a
        // cancelled run leaves a partial cloud behind, and the UI must see
        // the state of the run that just ended — never the state of an
        // older one.
        publishResult();
    }
    return ok;
}

bool ReconstructionService::isRunning() const {
    return m_running;
}

void ReconstructionService::stopReconstruction() {
    // Hủy cooperative: pipeline tự thoát tại checkpoint kế tiếp rồi worker
    // emit finished(false) → quit() → dọn dẹp → reconstructFinished(false).
    // Không block thread gọi (UI) và không terminate() giữa chừng — terminate
    // có thể bỏ m_mutex ở trạng thái khóa vĩnh viễn (worker đang giữ khi chạy
    // pipeline), khiến mọi lệnh sau đó deadlock.
    m_cancelRequested = true;
}

QStringList ReconstructionService::getImageList() const {
    // The published copy is read under the published lock only: the pipeline
    // lock is held by the worker for the whole reconstruction, so taking it
    // here would freeze the UI for the entire run.
    std::lock_guard pub(m_publishedMutex);
    return m_imageList;
}

std::shared_ptr<const PointCloudSnapshot> ReconstructionService::pointCloudSnapshot() const {
    // The PUBLISHED state only: the pipeline lock is held by the worker for
    // the whole reconstruction, so reading the pipeline here would freeze the
    // UI for the entire run. The snapshot is built and published by the
    // worker when the run ends (publishResult).
    std::lock_guard pub(m_publishedMutex);
    return m_snapshot;
}

size_t ReconstructionService::pointCount() const {
    // Published together with the snapshot — the UI never touches the
    // pipeline, not even to count its points.
    std::lock_guard pub(m_publishedMutex);
    return m_publishedPointCount;
}

bool ReconstructionService::hasResult() const {
    std::lock_guard pub(m_publishedMutex);
    return m_snapshot != nullptr;
}

void ReconstructionService::processPointCloud() {
    std::unique_lock<std::shared_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        qWarning() << "ReconstructionService: the pipeline is busy — the point cloud was not processed.";
        return;
    }
    if (m_running) return;
    m_pipeline->processPointCloud();
    // The processed cloud replaces the published result.
    publishResult();
}

void ReconstructionService::publishResult() {
    // The caller holds m_mutex (the pipeline lock) — the pipeline data is
    // read under it. The publication takes the published lock, so the lock
    // order stays m_mutex → m_publishedMutex.
    std::lock_guard pub(m_publishedMutex);
    m_snapshot.reset();
    m_publishedPointCount = 0;
    if (!m_pipeline || !m_pipeline->hasResult()) return;

    m_snapshot = m_pipeline->buildSnapshot();
    m_publishedPointCount = m_snapshot->points.size();
}
