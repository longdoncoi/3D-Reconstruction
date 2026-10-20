#include "ReconstructionService.h"
#include "ReconstructionWorker.h"
#include <QDir>

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
            m_thread->wait();
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
    if (m_running) return;
    std::vector<QString> stdPaths;
    stdPaths.reserve(static_cast<size_t>(paths.size()));
    for (const QString& p : paths)
        stdPaths.push_back(QDir::toNativeSeparators(p));
    
    std::unique_lock lock(m_mutex);
    m_pipeline->setImages(stdPaths);
    m_snapshot.reset();
}

bool ReconstructionService::loadCameraParams(const QString& filePath) {
    if (m_running) return false;
    std::unique_lock lock(m_mutex);
    const bool ok = m_pipeline->loadCameraParams(filePath);
    if (ok) m_snapshot.reset();
    return ok;
}

void ReconstructionService::configure(int densityLevel, int maxFeatures, bool autoClean) {
    if (m_running) return;
    std::unique_lock lock(m_mutex);
    
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
        // before the worker starts mutating the pipeline.
        std::unique_lock lock(m_mutex);
        m_snapshot.reset();
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
    if (m_thread) {
        m_thread->wait();
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
    bool ok;
    {
        std::unique_lock lock(m_mutex);
        ok = m_pipeline->reconstruct(
            [this] { return m_cancelRequested.load(std::memory_order_relaxed); });
        m_lastError = ok ? QString() : m_pipeline->errorString();
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
    std::shared_lock lock(m_mutex);
    QStringList result;
    for (const QString& p : m_pipeline->getImages())
        result.append(p);
    return result;
}

std::shared_ptr<const PointCloudSnapshot> ReconstructionService::pointCloudSnapshot() const {
    std::unique_lock lock(m_mutex);
    if (m_running || !m_pipeline || !m_pipeline->hasResult())
        return {};

    if (!m_snapshot) {
        auto snapshot = std::make_shared<PointCloudSnapshot>();

        const auto& srcPoints = m_pipeline->getPointCloud();
        snapshot->points.reserve(srcPoints.size());
        for (const auto& p : srcPoints)
            snapshot->points.push_back({p.x, p.y, p.z});

        const auto& srcColors = m_pipeline->getPointColors();
        snapshot->colors.reserve(srcColors.size());
        for (const auto& c : srcColors)
            snapshot->colors.push_back({c[0], c[1], c[2]});

        m_snapshot = std::move(snapshot);
    }
    return m_snapshot;
}

size_t ReconstructionService::pointCount() const {
    if (m_running || !m_pipeline) return 0;
    std::shared_lock lock(m_mutex);
    return m_pipeline->pointCount();
}

bool ReconstructionService::hasResult() const {
    if (m_running || !m_pipeline) return false;
    std::shared_lock lock(m_mutex);
    return m_pipeline->hasResult();
}

void ReconstructionService::processPointCloud() {
    if (m_running) return;
    std::unique_lock lock(m_mutex);
    m_pipeline->processPointCloud();
    m_snapshot.reset();
}
