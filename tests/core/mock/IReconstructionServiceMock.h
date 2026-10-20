#ifndef IRECONSTRUCTIONSERVICEMOCK_H
#define IRECONSTRUCTIONSERVICEMOCK_H

#include "core/IReconstructionService.h"

/**
 * @brief Mock implementation of IReconstructionService for architecture boundary tests.
 * Does not require gmock — proves plugin can depend only on interface.
 */
class MockReconstructionService : public IReconstructionService {
public:
    MockReconstructionService() = default;
    void setImages(const QStringList&) override {}
    bool loadCameraParams(const QString&) override { return true; }
    void configure(int, int, bool) override {}
    bool reconstruct() override { return true; }
    bool isRunning() const override { return false; }
    void stopReconstruction() override {}
    QStringList getImageList() const override { return {}; }
    std::shared_ptr<const PointCloudSnapshot> pointCloudSnapshot() const override { return {}; }
    size_t pointCount() const override { return 0; }
    bool hasResult() const override { return false; }
    void processPointCloud() override {}
};

#endif
