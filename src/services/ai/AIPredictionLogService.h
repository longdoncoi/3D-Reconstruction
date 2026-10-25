#ifndef AIPREDICTIONLOGSERVICE_H
#define AIPREDICTIONLOGSERVICE_H

#include "IAIPredictionLogService.h"
#include "Global.h"

#include <QObject>
#include <QThreadPool>

#include <functional>

class IAppPaths;

/**
 * @brief The AppCore implementation of IAIPredictionLogService.
 *
 * Owns a single-thread pool so that:
 *  - the write never runs on the GUI thread, and
 *  - the worker is joined in the destructor, so no task outlives the
 *    service it uses.
 */
class APP_EXPORT AIPredictionLogService : public QObject, public IAIPredictionLogService {
    Q_OBJECT
public:
    explicit AIPredictionLogService(IAppPaths *paths, QObject *parent = nullptr);
    ~AIPredictionLogService() override;

    void logPrediction(const QString &mode, const QString &sourceImagePath,
                       const cv::Mat &resultImage) override;

    /// The platform writer: returns false when the image could not be encoded.
    using Writer = std::function<bool(const QString &path, const cv::Mat &image)>;

    /**
     * @brief Test seam for the writer.
     *
     * The worker path is the one that must be proven to CONTAIN a throwing
     * encoder (`cv::imwrite` raises `cv::Exception` on a full disk, a path
     * OpenCV cannot encode, ...). Without a seam the only way to exercise it
     * would be to depend on the host filesystem's failure modes.
     */
    void setWriterForTesting(Writer writer);

private:
    IAppPaths *m_paths = nullptr; ///< injected, not owned
    QThreadPool m_pool;           ///< owned; joined in the destructor
    Writer m_writer;              ///< defaults to cv::imwrite
};

#endif // AIPREDICTIONLOGSERVICE_H
