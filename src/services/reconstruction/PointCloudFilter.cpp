#include "PointCloudFilter.h"
#include <QDebug>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct VoxelKey {
    int x;
    int y;
    int z;

    bool operator==(const VoxelKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey &key) const
    {
        const std::uint64_t x = static_cast<std::uint32_t>(key.x);
        const std::uint64_t y = static_cast<std::uint32_t>(key.y);
        const std::uint64_t z = static_cast<std::uint32_t>(key.z);
        std::uint64_t hash = x * 73856093u;
        hash ^= y * 19349663u;
        hash ^= z * 83492791u;
        return static_cast<std::size_t>(hash);
    }
};

struct VoxelAccumulator {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double blue = 0.0;
    double green = 0.0;
    double red = 0.0;
    int count = 0;
};

// ─── helpers ────────────────────────────────────────────────────────────────

static pcl::PointCloud<pcl::PointXYZ>::Ptr buildXYZCloud(
    const std::vector<cv::Point3f> &pts)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        cloud->points[i].x = pts[i].x;
        cloud->points[i].y = pts[i].y;
        cloud->points[i].z = pts[i].z;
    }
    cloud->width  = (uint32_t)pts.size();
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static void applyIndices(const std::vector<int> &idx,
                         std::vector<cv::Point3f> &pts,
                         std::vector<cv::Vec3b>   &cols)
{
    std::vector<cv::Point3f> np;
    std::vector<cv::Vec3b>   nc;
    np.reserve(idx.size());
    nc.reserve(idx.size());
    for (int i : idx) { np.push_back(pts[i]); nc.push_back(cols[i]); }
    pts.swap(np);
    cols.swap(nc);
}

// ─── public methods ─────────────────────────────────────────────────────────

void PointCloudFilter::statisticalOutlier(std::vector<cv::Point3f> &pts,
                                           std::vector<cv::Vec3b>   &cols,
                                           float meanK,
                                           float stdDevMulThresh)
{
    if (pts.empty()) return;
    int k = std::max(1, (int)meanK);
    if ((int)pts.size() <= k) { qDebug() << "SOR: too few points, skipping."; return; }
    size_t n = pts.size();
    auto cloud = buildXYZCloud(pts);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(k);
    sor.setStddevMulThresh(stdDevMulThresh);
    std::vector<int> kept;
    sor.filter(kept);
    applyIndices(kept, pts, cols);
    qDebug() << "SOR: kept" << pts.size() << "/" << n;
}

void PointCloudFilter::radiusOutlier(std::vector<cv::Point3f> &pts,
                                      std::vector<cv::Vec3b>   &cols,
                                      float radius,
                                      int   minNeighbors)
{
    if (pts.empty() || radius <= 0.f || minNeighbors < 1) return;
    size_t n = pts.size();
    auto cloud = buildXYZCloud(pts);
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(minNeighbors);
    std::vector<int> kept;
    ror.filter(kept);
    applyIndices(kept, pts, cols);
    qDebug() << "ROR: kept" << pts.size() << "/" << n;
}

void PointCloudFilter::adaptiveRadiusOutlier(std::vector<cv::Point3f> &pts,
                                              std::vector<cv::Vec3b> &cols,
                                              float radiusMultiplier,
                                              int minNeighbors)
{
    if (pts.size() < 16 || radiusMultiplier <= 0.f || minNeighbors < 1) return;
    auto cloud = buildXYZCloud(pts);
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    std::vector<float> distances;
    distances.reserve(pts.size());
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    for (const auto &point : cloud->points) {
        if (tree.nearestKSearch(point, 2, indices, squaredDistances) == 2 && squaredDistances[1] > 0.f)
            distances.push_back(std::sqrt(squaredDistances[1]));
    }
    if (distances.empty()) return;
    const auto middle = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), middle, distances.end());
    const float radius = *middle * radiusMultiplier;
    if (!std::isfinite(radius) || radius <= 0.f) return;
    qDebug() << "Adaptive ROR radius:" << radius;
    radiusOutlier(pts, cols, radius, minNeighbors);
}

void PointCloudFilter::voxelGrid(std::vector<cv::Point3f> &pts,
                                  std::vector<cv::Vec3b>   &cols,
                                  float leafSize)
{
    if (pts.empty() || leafSize <= 0.f) return;
    size_t n = pts.size();
    if (cols.size() != pts.size()) {
        qWarning() << "VoxelGrid: color count mismatch; filling missing colors."
                   << cols.size() << "/" << pts.size();
        cols.resize(pts.size(), cv::Vec3b(128, 128, 128));
    }

    // Compute bbox to avoid PCL integer overflow
    float xmin = pts[0].x, xmax = xmin;
    float ymin = pts[0].y, ymax = ymin;
    float zmin = pts[0].z, zmax = zmin;
    for (const auto &p : pts) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }
    float range = std::max({xmax - xmin, ymax - ymin, zmax - zmin});
    float leaf  = std::max(leafSize, range / 2048.0f);
    if (leaf != leafSize)
        qDebug() << "VoxelGrid: leaf adjusted to" << leaf << "to prevent overflow.";

    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels;
    voxels.reserve(n);
    for (size_t index = 0; index < n; ++index) {
        const cv::Point3f &point = pts[index];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;

        VoxelKey key{
            static_cast<int>(std::floor((point.x - xmin) / leaf)),
            static_cast<int>(std::floor((point.y - ymin) / leaf)),
            static_cast<int>(std::floor((point.z - zmin) / leaf))
        };

        VoxelAccumulator &accumulator = voxels[key];
        accumulator.x += point.x;
        accumulator.y += point.y;
        accumulator.z += point.z;
        accumulator.blue += cols[index][0];
        accumulator.green += cols[index][1];
        accumulator.red += cols[index][2];
        ++accumulator.count;
    }

    pts.clear(); cols.clear();
    pts.reserve(voxels.size()); cols.reserve(voxels.size());
    for (const auto &entry : voxels) {
        const VoxelAccumulator &accumulator = entry.second;
        if (accumulator.count <= 0)
            continue;

        const double invCount = 1.0 / accumulator.count;
        pts.push_back(cv::Point3f(static_cast<float>(accumulator.x * invCount),
                                  static_cast<float>(accumulator.y * invCount),
                                  static_cast<float>(accumulator.z * invCount)));
        cols.push_back(cv::Vec3b(
            static_cast<uchar>(std::clamp(std::lround(accumulator.blue * invCount), 0l, 255l)),
            static_cast<uchar>(std::clamp(std::lround(accumulator.green * invCount), 0l, 255l)),
            static_cast<uchar>(std::clamp(std::lround(accumulator.red * invCount), 0l, 255l))));
    }
    qDebug() << "VoxelGrid: kept" << pts.size() << "/" << n;
}

void PointCloudFilter::densityFilter(std::vector<cv::Point3f> &pts,
                                      std::vector<cv::Vec3b>   &cols,
                                      float radius, int minNeighbors)
{
    if (pts.empty()) return;
    size_t n = pts.size();
    std::vector<bool> keep(n, false);
    for (size_t i = 0; i < n; ++i) {
        int cnt = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            float dx = pts[i].x - pts[j].x;
            float dy = pts[i].y - pts[j].y;
            float dz = pts[i].z - pts[j].z;
            if (std::sqrt(dx*dx + dy*dy + dz*dz) < radius && ++cnt >= minNeighbors) break;
        }
        if (cnt >= minNeighbors) keep[i] = true;
    }
    std::vector<cv::Point3f> np; std::vector<cv::Vec3b> nc;
    for (size_t i = 0; i < n; ++i)
        if (keep[i]) { np.push_back(pts[i]); nc.push_back(cols[i]); }
    qDebug() << "DensityFilter: kept" << np.size() << "/" << n;
    pts.swap(np); cols.swap(nc);
}
