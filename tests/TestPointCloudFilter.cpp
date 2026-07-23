#include "PointCloudFilter.h"
#include <QtTest>
#include <opencv2/core.hpp>

class TestPointCloudFilter : public QObject {
  Q_OBJECT

private slots:
  void testStatisticalOutlier() {
    std::vector<cv::Point3f> pts;
    std::vector<cv::Vec3b> cols;

    // Generate 100 close points (inliers)
    for (int i = 0; i < 100; ++i) {
      pts.push_back(cv::Point3f(i * 0.01f, i * 0.01f, 1.0f));
      cols.push_back(cv::Vec3b(255, 255, 255));
    }

    // Add 5 extreme outliers
    for (int i = 0; i < 5; ++i) {
      pts.push_back(cv::Point3f(10.0f + i, 10.0f + i, 10.0f + i));
      cols.push_back(cv::Vec3b(255, 0, 0));
    }

    QCOMPARE(pts.size(), 105);

    // Run SOR
    PointCloudFilter::statisticalOutlier(pts, cols, 10, 1.0f);

    // Outliers should be removed, leaving roughly 100 points
    QVERIFY(pts.size() >= 95 && pts.size() <= 100);
    QCOMPARE(pts.size(), cols.size());
  }

  void testVoxelGrid() {
    std::vector<cv::Point3f> pts;
    std::vector<cv::Vec3b> cols;

    // Add multiple points very close to each other (within 0.1)
    pts.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
    pts.push_back(cv::Point3f(0.01f, 0.01f, 0.0f));
    pts.push_back(cv::Point3f(0.02f, 0.02f, 0.0f));
    cols.push_back(cv::Vec3b(255, 0, 0));
    cols.push_back(cv::Vec3b(255, 0, 0));
    cols.push_back(cv::Vec3b(255, 0, 0));

    // Add points far away
    pts.push_back(cv::Point3f(1.0f, 1.0f, 1.0f));
    cols.push_back(cv::Vec3b(0, 255, 0));

    QCOMPARE(pts.size(), 4);

    // Use large leaf size so all first 3 points merge into 1
    PointCloudFilter::voxelGrid(pts, cols, 0.1f);

    // Should be left with exactly 2 points (one for origin area, one for
    // (1,1,1))
    QCOMPARE(pts.size(), 2);
    QCOMPARE(pts.size(), cols.size());
  }
};

QTEST_GUILESS_MAIN(TestPointCloudFilter)
#include "TestPointCloudFilter.moc"