#ifndef POINTCLOUDDATA_H
#define POINTCLOUDDATA_H

#include <vector>

/**
 * @brief ABI-neutral point cloud data — no OpenCV dependency.
 * Used by IReconstructionService and ISceneService so plugin consumers
 * never include opencv2/core.hpp.
 */
struct Point3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct PointColor {
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
};

/**
 * @brief Immutable point-cloud result shared across the plugin ABI.
 *
 * Points and colours are kept in one object so a consumer can never observe
 * two different reconstruction generations (the old two-call API allowed
 * that). The service caches the snapshot after a run and hands out a
 * shared_ptr; readers keep the data alive without copying it.
 */
struct PointCloudSnapshot {
    std::vector<Point3D> points;
    std::vector<PointColor> colors;
};

#endif // POINTCLOUDDATA_H
