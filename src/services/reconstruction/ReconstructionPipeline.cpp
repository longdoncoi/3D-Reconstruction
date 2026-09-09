#include "ReconstructionPipeline.h"
#include "CameraParamsParser.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "PoseEstimator.h"
#include "Triangulator.h"
#include "PointCloudFilter.h"
#include "AppConstants.h"

#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QElapsedTimer>
#include <QVector>
#include <QtConcurrent/QtConcurrent>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>

#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/io.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <set>

#include <map>
#include <unordered_map>

namespace {
constexpr const char *kReconstructionCacheFileName = "recon_cache_quality_v3.ply";

std::vector<cv::DMatch> filterMatchesByFundamental(
    const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    double ransacThreshold)
{
    if (matches.size() < 8 || ransacThreshold <= 0.0)
        return matches;

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(matches.size());
    points2.reserve(matches.size());
    for (const auto &match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat inlierMask;
    cv::Mat fundamental = cv::findFundamentalMat(
        points1, points2, cv::FM_RANSAC, ransacThreshold, 0.995, inlierMask);
    if (fundamental.empty() || inlierMask.empty())
        return std::vector<cv::DMatch>();

    std::vector<cv::DMatch> inliers;
    inliers.reserve(matches.size());
    for (int i = 0; i < inlierMask.rows; ++i) {
        if (inlierMask.at<uchar>(i))
            inliers.push_back(matches[i]);
    }

    return inliers.size() >= 8 ? inliers : std::vector<cv::DMatch>();
}
}

// ─── Constructor ─────────────────────────────────────────────────────────────

ReconstructionPipeline::ReconstructionPipeline() {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    K_fallback = (cv::Mat_<double>(3, 3) << 1500.0, 0.0, 300.0,
                  0.0, 1500.0, 250.0,
                  0.0, 0.0, 1.0);
    distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
}

ReconstructionPipeline::~ReconstructionPipeline() {}

void ReconstructionPipeline::setConfig(const ReconstructionConfig &cfg) {
    m_config = cfg;
}

// ─── setImages ───────────────────────────────────────────────────────────────

void ReconstructionPipeline::setImages(const std::vector<QString> &imagePaths) {
    imageFiles = imagePaths;
    images.clear();
    for (const auto &path : imagePaths) {
        cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
        if (!img.empty()) images.push_back(img);
        else qWarning() << "Cannot read image:" << path;
    }
    qDebug() << "Loaded" << images.size() << "images";
}

// ─── loadCameraParams ────────────────────────────────────────────────────────
//
//  FORMAT A (generic):   first line = N,  then N lines: imageName K(9) R(9) t(3)
//  FORMAT B (Middlebury): no count header, tokens: [...] imageName K(9) R(9) t(3)

bool ReconstructionPipeline::loadCameraParams(const QString &paramsFilePath) {
    hasGroundTruthParams = CameraParamsParser::loadFromFile(paramsFilePath, camParams);
    return hasGroundTruthParams;
}

// ─── Private helpers (thin wrappers to keep call-sites minimal) ──────────────

void ReconstructionPipeline::extractFeatures(int idx) {
    FeatureExtractor::extract(images[idx], m_config, keypoints[idx], descriptors[idx]);
}

void ReconstructionPipeline::matchFeatures(int idx1, int idx2,
                                           std::vector<cv::DMatch> &goodMatches) {
    FeatureMatcher::match(descriptors[idx1], descriptors[idx2], m_config, goodMatches);
}

bool ReconstructionPipeline::estimatePoseFromMatches(const std::vector<cv::Point2f> &pts1,
                                                     const std::vector<cv::Point2f> &pts2,
                                                     cv::Mat &R, cv::Mat &t) {
    return PoseEstimator::estimatePose(pts1, pts2, K_fallback, R, t,
                                       AppConstants::Reconstruction::MIN_INLIERS_FOR_POSE);
}

void ReconstructionPipeline::doTriangulate(const cv::Mat &P0, const cv::Mat &P1,
                                           const std::vector<cv::Point2f> &pts0,
                                           const std::vector<cv::Point2f> &pts1,
                                           std::vector<cv::Point3f> &outPts) {
    Triangulator::triangulate(P0, P1, pts0, pts1, outPts);
}

double ReconstructionPipeline::computeReprojectionError(const cv::Mat &P,
                                                         const cv::Point3f &pt3d,
                                                         const cv::Point2f &pt2d) {
    return PoseEstimator::reprojectionError(P, pt3d, pt2d);
}


// ─── processPointCloud ───────────────────────────────────────────────────────

void ReconstructionPipeline::processPointCloud() {
    if (points3D.empty()) return;
    qDebug() << "Post-processing. Initial points:" << points3D.size();

    const auto &f = m_config.filter;
    if (m_usedTrackBasedGroundTruth) {
        qDebug() << "Post-processing (GT track). Initial points:" << points3D.size();

        // B1: Adaptive ROR trước - loại bụi cô lập, giữ cột mỏng
        // Log 11662 của bạn: radius = 0.00433567, kept 13838/14819
        float mult = f.rorRadiusMultiplierTrack > 0 ? f.rorRadiusMultiplierTrack : 6.0f;
        int minNei = f.rorMinNeighborsTrack > 0 ? f.rorMinNeighborsTrack : 4;
        PointCloudFilter::adaptiveRadiusOutlier(points3D, colors, mult, minNei);
        if (points3D.empty()) { qWarning() << "No points after adaptive ROR."; return; }

        // B2: SOR - loại điểm bay lơ lửng, để 1.2f là cân bằng nhất
        float sorK = f.sorMeanKTrack > 0 ? f.sorMeanKTrack : 50.0f;
        float sorStd = f.sorStdDevMulTrack > 0 ? f.sorStdDevMulTrack : 1.2f;
        PointCloudFilter::statisticalOutlier(points3D, colors, sorK, sorStd);
        if (points3D.empty()) { qWarning() << "No points after SOR."; return; }

        // B3: VoxelGrid - giữ chi tiết, không *0.5 nữa
        float leaf = f.voxelLeafSizeTrack > 0 ? f.voxelLeafSizeTrack : 0.003f;

        PointCloudFilter::voxelGrid(points3D, colors, leaf);
    } else {
        PointCloudFilter::statisticalOutlier(points3D, colors, f.sorMeanK, f.sorStdDevMul);
        if (points3D.empty()) { qWarning() << "No points after SOR."; return; }
        
        // The reconstruction scale is arbitrary here; derive the radius from
        // nearest-neighbour spacing instead of a fixed world-space threshold.
        PointCloudFilter::adaptiveRadiusOutlier(points3D, colors, 3.0f, f.rorMinNeighbors);
        if (points3D.empty()) { qWarning() << "No points after adaptive ROR."; return; }

        PointCloudFilter::voxelGrid(points3D, colors, f.voxelLeafSize);
    }

    qDebug() << "After post-processing:" << points3D.size() << "points";
}

void ReconstructionPipeline::filterFarOutliers(float sigma)
{
    if (points3D.empty()) return;
    cv::Point3f centroid(0,0,0);
    for (auto &p: points3D){ centroid.x+=p.x; centroid.y+=p.y; centroid.z+=p.z; }
    centroid.x/=points3D.size(); centroid.y/=points3D.size(); centroid.z/=points3D.size();

    std::vector<float> dists; dists.reserve(points3D.size());
    for (auto &p: points3D){
        float dx=p.x-centroid.x, dy=p.y-centroid.y, dz=p.z-centroid.z;
        dists.push_back(sqrtf(dx*dx+dy*dy+dz*dz));
    }
    float mean=0; for(auto d:dists) mean+=d; mean/=dists.size();
    float var=0; for(auto d:dists) var+=(d-mean)*(d-mean); var/=dists.size();
    float stddev = sqrtf(var);
    float thresh = mean + sigma*stddev; // sigma=2.5 là vừa cho templeRing

    std::vector<cv::Point3f> p2; p2.reserve(points3D.size());
    std::vector<cv::Vec3b> c2; c2.reserve(colors.size());
    for(size_t i=0;i<points3D.size();++i){
        if(dists[i] <= thresh){ p2.push_back(points3D[i]); c2.push_back(colors[i]); }
    }
    qDebug() << "Far outlier filter:" << points3D.size() << "->" << p2.size()
             << " thresh" << thresh << " mean" << mean;
    points3D.swap(p2); colors.swap(c2);
}

std::vector<cv::Point3f> ReconstructionPipeline::densifyPointCloudMLS(
    const std::vector<cv::Point3f>& pts,
    const std::vector<cv::Vec3b>& cols,
    int targetCount)
{
    if (pts.empty()) return {};
    qDebug() << "MVS Densify (KNN Interpolate). Input:" << pts.size() << "target:" << targetCount;

    // Build KdTree cho cloud gốc
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->resize(pts.size());
    for (size_t i=0;i<pts.size();++i) (*cloud)[i] = pcl::PointXYZ(pts[i].x, pts[i].y, pts[i].z);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    int factor = std::max(1, targetCount / (int)pts.size()); // 2500000/11725 = 213
    std::vector<cv::Point3f> outPts; outPts.reserve(targetCount);
    std::vector<cv::Vec3b> outCols; outCols.reserve(targetCount);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.0005f); // nhiễu 0.5mm

    std::vector<int> indices(6);
    std::vector<float> sqDist(6);

    for (size_t i=0;i<pts.size();++i){
        outPts.push_back(pts[i]);
        outCols.push_back(cols[i]);

        if (kdtree.nearestKSearch((*cloud)[i], 6, indices, sqDist) < 2) continue;

        for (int f=0; f < factor-1; ++f){
            int j = indices[1 + (rng() % 5)]; // lấy ngẫu nhiên 1 trong 5 hàng xóm gần
            float t = dist(rng);
            cv::Point3f p;
            p.x = pts[i].x * (1-t) + pts[j].x * t + noise(rng);
            p.y = pts[i].y * (1-t) + pts[j].y * t + noise(rng);
            p.z = pts[i].z * (1-t) + pts[j].z * t + noise(rng);

            // nội suy màu
            cv::Vec3b c;
            c[0] = (uchar)(cols[i][0]*(1-t) + cols[j][0]*t);
            c[1] = (uchar)(cols[i][1]*(1-t) + cols[j][1]*t);
            c[2] = (uchar)(cols[i][2]*(1-t) + cols[j][2]*t);

            outPts.push_back(p);
            outCols.push_back(c);
            if ((int)outPts.size() >= targetCount) break;
        }
        if ((int)outPts.size() >= targetCount) break;
    }

    qDebug() << "MVS Densify done:" << outPts.size();
    points3D = outPts;
    colors = outCols;
    return outPts;
}

pcl::PolygonMesh ReconstructionPipeline::poissonMeshing(
    const std::vector<cv::Point3f>& densePts,
    const std::vector<cv::Vec3b>& denseCols,
    int poissonDepth)
{
    qDebug() << "Poisson Input:" << densePts.size() << "depth:" << poissonDepth;

    // Tính bounding box của point cloud gốc để lát cắt vòm
    cv::Point3f minPt(1e9,1e9,1e9), maxPt(-1e9,-1e9,-1e9);
    for(auto &p: densePts){
        minPt.x = std::min(minPt.x, p.x); minPt.y = std::min(minPt.y, p.y); minPt.z = std::min(minPt.z, p.z);
        maxPt.x = std::max(maxPt.x, p.x); maxPt.y = std::max(maxPt.y, p.y); maxPt.z = std::max(maxPt.z, p.z);
    }
    float diag = sqrtf((maxPt.x-minPt.x)*(maxPt.x-minPt.x) + (maxPt.y-minPt.y)*(maxPt.y-minPt.y) + (maxPt.z-minPt.z)*(maxPt.z-minPt.z));
    float expand = diag * 0.08f; // cho phép nở 8%
    minPt.x-=expand; minPt.y-=expand; minPt.z-=expand;
    maxPt.x+=expand; maxPt.y+=expand; maxPt.z+=expand;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->resize(densePts.size());
    for (size_t i=0;i<densePts.size();++i){
        (*cloud)[i].x = densePts[i].x; (*cloud)[i].y = densePts[i].y; (*cloud)[i].z = densePts[i].z;
        (*cloud)[i].r = denseCols[i][2]; (*cloud)[i].g = denseCols[i][1]; (*cloud)[i].b = denseCols[i][0];
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::PointNormal> ne;
    ne.setInputCloud(cloud);
    ne.setKSearch(30);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    ne.setSearchMethod(tree);
    ne.compute(*cloudNormals);
    pcl::copyPointCloud(*cloud, *cloudNormals);

    // Orient normal về centroid của cloud, không về 0,0,0
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    for (auto &p : cloudNormals->points){
        Eigen::Vector3f n(p.normal_x, p.normal_y, p.normal_z);
        Eigen::Vector3f vp(centroid[0]-p.x, centroid[1]-p.y, centroid[2]-p.z);
        if (n.dot(vp) < 0){ p.normal_x*=-1; p.normal_y*=-1; p.normal_z*=-1; }
    }

    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setDepth(poissonDepth);
    poisson.setInputCloud(cloudNormals);
    poisson.setIsoDivide(8);
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    // --- TRIM: cắt hết đỉnh Poisson nằm ngoài bounding box gốc ---
    pcl::PointCloud<pcl::PointXYZRGB> meshCloud;
    pcl::fromPCLPointCloud2(mesh.cloud, meshCloud);
    pcl::PointCloud<pcl::PointXYZRGB> trimmed;
    for(auto &pt: meshCloud.points){
        if(pt.x>=minPt.x && pt.x<=maxPt.x && pt.y>=minPt.y && pt.y<=maxPt.y && pt.z>=minPt.z && pt.z<=maxPt.z){
            trimmed.push_back(pt);
        }
    }
    qDebug() << "Poisson trimmed:" << meshCloud.size() << "->" << trimmed.size() << " (removed wing)";
    pcl::toPCLPointCloud2(trimmed, mesh.cloud);

    qDebug() << "Poisson done. Vertices:" << mesh.cloud.width << "Faces:" << mesh.polygons.size();
    return mesh;
}

bool ReconstructionPipeline::projectWithP(const cv::Point3f& pw, const CameraParams& cam, cv::Point2f& uv)
{
    if (cam.P.empty() || cam.P.rows!=3 || cam.P.cols!=4) return false;
    cv::Mat X = (cv::Mat_<double>(4,1) << pw.x, pw.y, pw.z, 1.0);
    cv::Mat x = cam.P * X; // 3x1 double
    double w = x.at<double>(2);
    if (w <= 1e-6) return false;
    uv.x = float(x.at<double>(0) / w);
    uv.y = float(x.at<double>(1) / w);
    return true;
}

void ReconstructionPipeline::textureFromImages(
    std::vector<cv::Point3f>& pts,
    std::vector<cv::Vec3b>& cols,
    const std::vector<cv::Mat>& images,
    const std::vector<CameraParams>& cams)
{
    qDebug() << "Texturing point cloud from" << images.size() << "images (best-view)...";
    int noHit = 0;
    for (size_t i=0;i<pts.size();++i){
        float bestScore = -1e9f;
        cv::Vec3b bestCol(0,0,0);
        bool found = false;
        for (size_t j=0;j<images.size() && j<cams.size();++j){
            cv::Point2f uv;
            if (!projectWithP(pts[i], cams[j], uv)) continue;
            if (uv.x < 2 || uv.y < 2 || uv.x >= images[j].cols-2 || uv.y >= images[j].rows-2) continue;
            cv::Vec3b c = images[j].at<cv::Vec3b>((int)uv.y,(int)uv.x);
            if (c[0] < 5 && c[1] < 5 && c[2] < 5) continue; // bỏ nền đen templeRing
            float cx = images[j].cols*0.5f, cy = images[j].rows*0.5f;
            float score = -((uv.x-cx)*(uv.x-cx) + (uv.y-cy)*(uv.y-cy)); // gần tâm = tốt
            if (score > bestScore){
                bestScore = score;
                bestCol = c;
                found = true;
            }
        }
        if (found) cols[i] = bestCol;
        else noHit++;
    }
    qDebug() << "Texturing done: noHit =" << noHit << "/" << pts.size();
}

void ReconstructionPipeline::textureMeshFromImages(
    pcl::PolygonMesh& mesh,
    const std::vector<cv::Mat>& images,
    const std::vector<CameraParams>& cams)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    qDebug() << "Texturing mesh best-view Vertices:" << cloud.size();

    for (auto &pt : cloud.points){
        cv::Point3f pw(pt.x, pt.y, pt.z);
        float bestScore = -1e9f;
        cv::Vec3b bestCol(0,0,0);
        bool found = false;
        for (size_t j=0;j<images.size() && j<cams.size();++j){
            cv::Point2f uv;
            if (!projectWithP(pw, cams[j], uv)) continue;
            if (uv.x < 2 || uv.y < 2 || uv.x >= images[j].cols-2 || uv.y >= images[j].rows-2) continue;
            cv::Vec3b c = images[j].at<cv::Vec3b>((int)uv.y,(int)uv.x);
            if (c[0] < 5 && c[1] < 5 && c[2] < 5) continue;
            float cx = images[j].cols*0.5f, cy = images[j].rows*0.5f;
            float score = -((uv.x-cx)*(uv.x-cx) + (uv.y-cy)*(uv.y-cy));
            if (score > bestScore){
                bestScore = score;
                bestCol = c;
                found = true;
            }
        }
        if (found){ pt.b = bestCol[0]; pt.g = bestCol[1]; pt.r = bestCol[2]; }
    }

    pcl::toPCLPointCloud2(cloud, mesh.cloud);
    QString out = QFileInfo(imageFiles[0]).absolutePath() + "/temple_final_textured_2M.ply";
    pcl::io::savePLYFileBinary(out.toStdString(), mesh);
    qDebug() << "Texture mesh saved to" << out << "Faces:" << mesh.polygons.size();
}


// ─── reconstructWithGroundTruth ──────────────────────────────────────────────

bool ReconstructionPipeline::reconstructWithGroundTruth() {
    qDebug() << "=== Mode: Track-based Multi-View Triangulation (ground-truth poses) ===";
    int N = (int)images.size();

    TrackBuilder builder;
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < (int)keypoints[i].size(); ++k)
            builder.registerObs(i, k);

    const int WINDOW = m_config.searchWindow;
    std::mutex mtx;
    std::vector<std::pair<int,int>> pairs;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < std::min(N, i + WINDOW + 1); ++j)
            pairs.push_back({i, j});

    cv::parallel_for_(cv::Range(0, (int)pairs.size()), [&](const cv::Range &range) {
        for (int r = range.start; r < range.end; ++r) {
            int i = pairs[r].first, j = pairs[r].second;
            std::vector<cv::DMatch> matches;
            matchFeatures(i, j, matches);
            matches = filterMatchesByFundamental(matches, keypoints[i], keypoints[j],
                                                 m_config.pairFundamentalRansacThreshold);
            if ((int)matches.size() < m_config.minMatches) continue;
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto &m : matches)
                builder.addMatch(i, m.queryIdx, j, m.trainIdx);
        }
    });

    auto tracks = builder.buildTracks(/*minObservations=*/2);   // ★ 2 thay vì 3
    qDebug() << "  Built" << tracks.size() << "multi-view tracks";

    MultiViewTriangulator::Params triParams;
    triParams.maxReprojError           = 2.5;   // ★ nới từ 1.2
    triParams.minTriangulationAngleDeg = 1.0;
    triParams.minObservations          = 2;

    points3D.clear(); colors.clear();
    std::mutex resultMtx;
    std::atomic<int> rejectedAngle{0}, rejectedTotal{0};

    cv::parallel_for_(cv::Range(0, (int)tracks.size()), [&](const cv::Range &range) {
        std::vector<cv::Point3f> lp; std::vector<cv::Vec3b> lc;
        for (int t = range.start; t < range.end; ++t) {
            FeatureTrack track = tracks[t];
            if (!MultiViewTriangulator::triangulateTrack(track, camParams, keypoints, triParams)) {
                rejectedTotal++;
                continue;
            }
            MultiViewTriangulator::refinePoint(track, camParams, keypoints);

            int imgIdx = track.observations[0].first, kpIdx = track.observations[0].second;
            cv::Point2f kp = keypoints[imgIdx][kpIdx].pt;
            int x = cvRound(kp.x), y = cvRound(kp.y);
            cv::Vec3b col(128,128,128);
            if (x >= 0 && y >= 0 && x < images[imgIdx].cols && y < images[imgIdx].rows)
                col = images[imgIdx].at<cv::Vec3b>(y, x);

            lp.push_back(track.point3D);
            lc.push_back(col);
        }
        std::lock_guard<std::mutex> lock(resultMtx);
        points3D.insert(points3D.end(), lp.begin(), lp.end());
        colors.insert(colors.end(), lc.begin(), lc.end());
    });

    qDebug() << "  Rejected tracks:" << rejectedTotal.load() << "/" << tracks.size();
    m_usedTrackBasedGroundTruth = true;  // ★ để processPointCloud() chọn đúng filter profile
    qDebug() << "=== Ground-truth track-based points:" << points3D.size() << "===";
    return !points3D.empty();
}

// ─── reconstructWithEstimatedPose ────────────────────────────────────────────

bool ReconstructionPipeline::reconstructWithEstimatedPose() {
    qDebug() << "=== Mode: ESTIMATED pose (incremental SfM via PnP resectioning) ===";
    int N = (int)images.size();

    K_fallback = PoseEstimator::estimateIntrinsics(images[0]);
    qDebug() << "  K: fx=" << K_fallback.at<double>(0,0)
             << "  cx=" << K_fallback.at<double>(0,2)
             << "  cy=" << K_fallback.at<double>(1,2);

    // ── Match cache: tránh chạy lại FLANN kNN + fundamental-matrix filter cho
    //    cùng một cặp ảnh nhiều lần trong lúc tìm base pair và trong mỗi vòng
    //    lặp incremental (nguồn chính gây runtime dài với chuỗi ảnh lớn).
    std::map<std::pair<int,int>, std::vector<cv::DMatch>> matchCache;
    auto matchFeaturesCached = [&](int a, int b, std::vector<cv::DMatch> &out) {
        auto it = matchCache.find({a, b});
        if (it != matchCache.end()) { out = it->second; return; }
        auto itRev = matchCache.find({b, a});
        if (itRev != matchCache.end()) {
            out.clear();
            out.reserve(itRev->second.size());
            for (const auto &m : itRev->second) {
                cv::DMatch sw = m;
                std::swap(sw.queryIdx, sw.trainIdx);
                out.push_back(sw);
            }
            matchCache[{a, b}] = out;
            return;
        }
        matchFeatures(a, b, out);
        out = filterMatchesByFundamental(out, keypoints[a], keypoints[b], m_config.pairFundamentalRansacThreshold);
        matchCache[{a, b}] = out;
    };

    // ── obsIndex: (imageIdx, keypointIdx) → index vào points3D. Cho biết,
    //    với bất kỳ ảnh nào, keypoint nào của nó đã tương ứng với một điểm 3D
    //    đã dựng — nền tảng để resection bằng PnP.
    std::unordered_map<int64_t, int> obsIndex;
    auto obsKey = [](int img, int kp) -> int64_t {
        return (int64_t)img * 100000 + kp;
    };

    // ── Base pair: quét TOÀN BỘ chuỗi ảnh (không giới hạn 20 ảnh đầu như
    //    trước) — giới hạn cũ có thể bỏ lỡ cặp ảnh tốt hơn nếu các ảnh đầu
    //    tiên có overlap kém (mờ, thiếu sáng, hoặc là điểm nối đầu/cuối quỹ
    //    đạo). Cửa sổ ±BASE_PAIR_WINDOW giữ chi phí ở mức O(N), rẻ hơn nhiều
    //    so với bước incremental phía sau.
    const int BASE_PAIR_WINDOW = 5;
    int    best_i = 0, best_j = 1;
    size_t bestInliers = 0;
    double bestRatio = 0.0;

    qDebug() << "  Finding base pair (full scan, window=" << BASE_PAIR_WINDOW << ")...";
    for (int i = 0; i < N - 1; ++i) {
        for (int jj = i + 1; jj <= std::min(N - 1, i + BASE_PAIR_WINDOW); ++jj) {
            std::vector<cv::DMatch> tmp;
            matchFeaturesCached(i, jj, tmp);
            if ((int)tmp.size() < 30) continue;

            std::vector<cv::Point2f> p1, p2;
            p1.reserve(tmp.size()); p2.reserve(tmp.size());
            for (const auto &m : tmp) {
                p1.push_back(keypoints[i][m.queryIdx].pt);
                p2.push_back(keypoints[jj][m.trainIdx].pt);
            }
            cv::Mat R_t, t_t, mask;
            cv::Mat E = cv::findEssentialMat(p1, p2, K_fallback, cv::RANSAC, 0.999, 1.0, mask);
            if (E.empty()) continue;
            int inl = cv::recoverPose(E, p1, p2, K_fallback, R_t, t_t, mask);
            double ratio = (double)inl / tmp.size();
            if (inl > (int)bestInliers &&
                inl >= AppConstants::Reconstruction::MIN_INLIERS_FOR_ESTIMATED_POSE &&
                ratio > 0.5)
            {
                bestInliers = inl; best_i = i; best_j = jj; bestRatio = ratio;
            }
        }
    }
    qDebug() << "  Base pair:" << best_i << "-" << best_j
             << " inliers=" << bestInliers << " ratio=" << bestRatio;

    if (bestInliers < (size_t)m_config.minMatches) {
        // Fallback chỉ được chấp nhận nếu vẫn đạt tối thiểu về mặt hình học
        // (inlier ratio > 0.35). Trước đây fallback chọn cặp thuần theo số
        // lượng match thô, có thể chấp nhận một Essential Matrix không ổn
        // định — làm lệch toàn bộ hệ tọa độ 3D dựng từ đó về sau.
        qWarning() << "No good base pair in primary pass — trying relaxed fallback...";
        size_t bestMatch = 0;
        int fb_i = best_i, fb_j = best_j;
        for (int i = 0; i < N - 1; ++i) {
            for (int jj = i + 1; jj <= std::min(N - 1, i + 3); ++jj) {
                std::vector<cv::DMatch> tmp;
                matchFeaturesCached(i, jj, tmp);
                if (tmp.size() > bestMatch) { bestMatch = tmp.size(); fb_i = i; fb_j = jj; }
            }
        }
        std::vector<cv::DMatch> tmp2; matchFeaturesCached(fb_i, fb_j, tmp2);
        std::vector<cv::Point2f> p1, p2;
        for (const auto &m : tmp2) {
            p1.push_back(keypoints[fb_i][m.queryIdx].pt);
            p2.push_back(keypoints[fb_j][m.trainIdx].pt);
        }
        cv::Mat Rt, tt, mask2;
        cv::Mat E2 = cv::findEssentialMat(p1, p2, K_fallback, cv::RANSAC, 0.999, 1.0, mask2);
        size_t fbInliers = 0;
        if (!E2.empty()) fbInliers = cv::recoverPose(E2, p1, p2, K_fallback, Rt, tt, mask2);
        double fbRatio = tmp2.empty() ? 0.0 : (double)fbInliers / tmp2.size();

        if (fbInliers < 10 || fbRatio <= 0.35) {
            qCritical() << "Base pair search failed (best fallback inliers=" << fbInliers
                        << " ratio=" << fbRatio << ") — reconstruction aborted.";
            return false;
        }
        best_i = fb_i; best_j = fb_j; bestInliers = fbInliers; bestRatio = fbRatio;
        qWarning() << "  Fallback base pair accepted:" << best_i << "-" << best_j
                   << " inliers=" << bestInliers << " ratio=" << bestRatio;
    }

    // Base pair pose (2-view — bắt buộc dùng Essential Matrix vì chưa có điểm 3D nào)
    std::vector<cv::DMatch> baseMatches;
    matchFeaturesCached(best_i, best_j, baseMatches);
    std::vector<cv::Point2f> bp1, bp2;
    for (const auto &m : baseMatches) {
        bp1.push_back(keypoints[best_i][m.queryIdx].pt);
        bp2.push_back(keypoints[best_j][m.trainIdx].pt);
    }
    cv::Mat R_base, t_base;
    if (!estimatePoseFromMatches(bp1, bp2, R_base, t_base)) {
        qCritical() << "estimatePoseFromMatches failed for base pair.";
        return false;
    }

    cv::Mat P0 = K_fallback * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat RT0; cv::hconcat(R_base, t_base, RT0);
    cv::Mat P1 = K_fallback * RT0;

    std::vector<cv::Point3f> basePts;
    doTriangulate(P0, P1, bp1, bp2, basePts);

    // Cheirality (depth dương ở CẢ HAI camera) + reprojection filter, đúng
    // với mọi quỹ đạo camera (kể cả quét orbital 360°) thay vì phụ thuộc
    // hướng world-Z của camera gốc.
    points3D.clear(); colors.clear(); obsIndex.clear();
    for (size_t k = 0; k < basePts.size() && k < baseMatches.size(); ++k) {
        const auto &pt = basePts[k];
        cv::Mat p4 = (cv::Mat_<double>(4,1) << pt.x, pt.y, pt.z, 1.0);
        double depth0 = cv::Mat(P0.row(2) * p4).at<double>(0);
        double depth1 = cv::Mat(P1.row(2) * p4).at<double>(0);
        if (depth0 <= 0.0 || depth1 <= 0.0) continue;
        if (computeReprojectionError(P0, pt, bp1[k]) > m_config.reprojectionErrorMax) continue;
        if (computeReprojectionError(P1, pt, bp2[k]) > m_config.reprojectionErrorMax) continue;

        int newIdx = (int)points3D.size();
        points3D.push_back(pt);
        cv::Point2f kp = keypoints[best_i][baseMatches[k].queryIdx].pt;
        int x = cvRound(kp.x), y = cvRound(kp.y);
        cv::Vec3b col(128, 128, 128);
        if (x >= 0 && y >= 0 && x < images[best_i].cols && y < images[best_i].rows)
            col = images[best_i].at<cv::Vec3b>(y, x);
        colors.push_back(col);

        obsIndex[obsKey(best_i, baseMatches[k].queryIdx)] = newIdx;
        obsIndex[obsKey(best_j, baseMatches[k].trainIdx)] = newIdx;
    }
    qDebug() << "  Base triangulated:" << points3D.size();
    if (points3D.empty()) {
        qCritical() << "Base pair produced no valid points after cheirality/reprojection filter.";
        return false;
    }

    // Incremental SfM — resection từng camera mới bằng solvePnPRansac dựa
    // trên cấu trúc 3D ĐÃ CÓ (không chain pose tương đối), giữ scale nhất
    // quán suốt toàn bộ chuỗi ảnh.
    struct PoseInfo { int imgIdx; cv::Mat P, R, t; };
    std::vector<PoseInfo> knownPoses;
    knownPoses.push_back({best_i, P0.clone(), cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F)});
    knownPoses.push_back({best_j, P1.clone(), R_base.clone(), t_base.clone()});
    std::set<int> processed = {best_i, best_j};

    int consecutiveFailures = 0;
    for (int iter = 0; iter < N - 2; ++iter) {
        auto findNextCamera = [&](bool limitWindow) {
            int bNew = -1;
            int bRef = -1;
            std::vector<cv::Point3f> bObjPts;
            std::vector<cv::Point2f> bImgPts;
            std::vector<int> bObjIndices;
            std::vector<int> bImgKpIndices;

            for (int idx = 0; idx < N; ++idx) {
                if (processed.count(idx)) continue;

                std::vector<cv::Point3f> objPts;
                std::vector<cv::Point2f> imgPts;
                std::vector<int> objIndices;
                std::vector<int> imgKpIndices;
                std::map<int, int> refMatchCount;
                std::set<int> usedKeypoints;

                for (const auto &kp : knownPoses) {
                    int refImg = kp.imgIdx;
                    int dist = std::abs(idx - refImg);
                    int ringDist = std::min(dist, N - dist);
                    if (limitWindow && ringDist > m_config.searchWindow) continue;

                    std::vector<cv::DMatch> matches;
                    matchFeaturesCached(idx, refImg, matches);
                    refMatchCount[refImg] = (int)matches.size();

                    for (const auto &m : matches) {
                        if (usedKeypoints.count(m.queryIdx)) continue;
                        auto it = obsIndex.find(obsKey(refImg, m.trainIdx));
                        if (it != obsIndex.end()) {
                            objPts.push_back(points3D[it->second]);
                            imgPts.push_back(keypoints[idx][m.queryIdx].pt);
                            objIndices.push_back(it->second);
                            imgKpIndices.push_back(m.queryIdx);
                            usedKeypoints.insert(m.queryIdx);
                        }
                    }
                }

                if (objPts.size() > bObjPts.size()) {
                    bNew = idx;
                    bObjPts = std::move(objPts);
                    bImgPts = std::move(imgPts);
                    bObjIndices = std::move(objIndices);
                    bImgKpIndices = std::move(imgKpIndices);

                    int maxMatches = 0;
                    for (const auto &pair : refMatchCount) {
                        if (pair.second > maxMatches) {
                            maxMatches = pair.second;
                            bRef = pair.first;
                        }
                    }
                }
            }
            return std::make_tuple(bNew, bRef, bObjPts, bImgPts, bObjIndices, bImgKpIndices);
        };

        int bestNew, bestRef;
        std::vector<cv::Point3f> bestObjPts;
        std::vector<cv::Point2f> bestImgPts;
        std::vector<int> bestObjIndices;
        std::vector<int> bestImgKpIndices;

        std::tie(bestNew, bestRef, bestObjPts, bestImgPts, bestObjIndices, bestImgKpIndices) = findNextCamera(true);
        if (bestNew < 0 || (int)bestObjPts.size() < AppConstants::Reconstruction::MIN_POINTS_FOR_PNP) {
            std::tie(bestNew, bestRef, bestObjPts, bestImgPts, bestObjIndices, bestImgKpIndices) = findNextCamera(false);
        }

        if (bestNew < 0 || (int)bestObjPts.size() < AppConstants::Reconstruction::MIN_POINTS_FOR_PNP) {
            qDebug() << "  No more images can be resectioned. Stopping.";
            break;
        }

        cv::Mat rvec, tvec;
        std::vector<int> pnpInliers;
        // Ngưỡng RANSAC dùng PNP_REPROJECTION_ERROR_PX (4.0px, đã định nghĩa
        // trong AppConstants — trước đây bị bỏ qua, hardcode 8.0px) để gần
        // với reprojectionErrorMax chấp nhận điểm triangulate, giảm khả năng
        // một pose "vừa đủ qua ải RANSAC" nhưng thực chất lệch bị chấp nhận
        // rồi lan sai số sang các bước resection tiếp theo.
        bool pnpOk = cv::solvePnPRansac(
            bestObjPts, bestImgPts, K_fallback, distCoeffs,
            rvec, tvec, false, 1000,
            (float)AppConstants::Reconstruction::PNP_REPROJECTION_ERROR_PX,
            0.999, pnpInliers, cv::SOLVEPNP_EPNP);

        if (!pnpOk || (int)pnpInliers.size() < AppConstants::Reconstruction::MIN_POINTS_FOR_PNP) {
            qDebug() << "  PnP failed for image" << bestNew
                     << "(inliers=" << pnpInliers.size() << "/" << bestObjPts.size() << "). Skipping.";
            processed.insert(bestNew);
            if (++consecutiveFailures >= 5) {
                qWarning() << "  5 consecutive resectioning failures — stopping early.";
                break;
            }
            continue;
        }
        consecutiveFailures = 0;

        std::vector<cv::Point3f> inlierObj; std::vector<cv::Point2f> inlierImg;
        inlierObj.reserve(pnpInliers.size()); inlierImg.reserve(pnpInliers.size());
        for (int idx : pnpInliers) {
            inlierObj.push_back(bestObjPts[idx]);
            inlierImg.push_back(bestImgPts[idx]);
            // Cập nhật obsIndex để ghi nhận rằng camera mới này đang nhìn thấy
            // các điểm 3D cũ — cực kỳ quan trọng để liên kết các camera tiếp theo.
            obsIndex[obsKey(bestNew, bestImgKpIndices[idx])] = bestObjIndices[idx];
        }
        cv::solvePnPRefineLM(inlierObj, inlierImg, K_fallback, distCoeffs, rvec, tvec);

        cv::Mat R_abs; cv::Rodrigues(rvec, R_abs);
        cv::Mat t_abs = tvec;
        cv::Mat RT_n; cv::hconcat(R_abs, t_abs, RT_n);
        cv::Mat P_new = K_fallback * RT_n;

        knownPoses.push_back({bestNew, P_new.clone(), R_abs.clone(), t_abs.clone()});
        processed.insert(bestNew);

        std::vector<cv::DMatch> triMatches;
        matchFeaturesCached(bestNew, bestRef, triMatches);

        cv::Mat P_ref;
        for (const auto &kp : knownPoses) if (kp.imgIdx == bestRef) { P_ref = kp.P; break; }

        std::vector<cv::Point2f> pNew, pRef;
        std::vector<cv::DMatch>  newObsMatches;
        for (const auto &m : triMatches) {
            if (obsIndex.count(obsKey(bestRef, m.trainIdx))) continue; // đã có điểm 3D rồi
            pNew.push_back(keypoints[bestNew][m.queryIdx].pt);
            pRef.push_back(keypoints[bestRef][m.trainIdx].pt);
            newObsMatches.push_back(m);
        }

        std::vector<cv::Point3f> newPts;
        doTriangulate(P_ref, P_new, pRef, pNew, newPts);

        int added = 0;
        for (size_t k = 0; k < newPts.size() && k < newObsMatches.size(); ++k) {
            const auto &pt = newPts[k];
            cv::Mat p4 = (cv::Mat_<double>(4,1) << pt.x, pt.y, pt.z, 1.0);
            double depthRef = cv::Mat(P_ref.row(2) * p4).at<double>(0);
            double depthNew = cv::Mat(P_new.row(2) * p4).at<double>(0);
            if (depthRef <= 0.0 || depthNew <= 0.0) continue;
            if (computeReprojectionError(P_ref, pt, pRef[k]) > m_config.reprojectionErrorMax) continue;
            if (computeReprojectionError(P_new, pt, pNew[k]) > m_config.reprojectionErrorMax) continue;

            int newIdx = (int)points3D.size();
            points3D.push_back(pt);
            cv::Point2f kp2 = keypoints[bestNew][newObsMatches[k].queryIdx].pt;
            int x = cvRound(kp2.x), y = cvRound(kp2.y);
            cv::Vec3b col(128, 128, 128);
            if (x >= 0 && y >= 0 && x < images[bestNew].cols && y < images[bestNew].rows)
                col = images[bestNew].at<cv::Vec3b>(y, x);
            colors.push_back(col);

            obsIndex[obsKey(bestNew, newObsMatches[k].queryIdx)] = newIdx;
            obsIndex[obsKey(bestRef,  newObsMatches[k].trainIdx)] = newIdx;
            ++added;
        }

        qDebug() << "  Image" << bestNew << "resectioned (ref=" << bestRef
                 << " inliers=" << pnpInliers.size() << "/" << bestObjPts.size() << ")"
                 << "added" << added << "pts. Total:" << points3D.size();
    }

    const double coverage = N > 0 ? (double)processed.size() / N : 0.0;
    qDebug() << "=== Estimated pose (PnP) raw points:" << points3D.size()
             << "| cameras resectioned:" << processed.size() << "/" << N
             << "(" << QString::number(coverage * 100.0, 'f', 1) << "%) ===";
    if (coverage < 0.5) {
        qWarning() << "Only resectioned" << processed.size() << "/" << N
                   << "images (" << QString::number(coverage * 100.0, 'f', 1)
                   << "%) — result may be incomplete or unreliable. "
                      "Check image overlap/order, or verify camera intrinsics estimate.";
    }
    return !points3D.empty();
}

// ─── reconstruct (public) ────────────────────────────────────────────────────

bool ReconstructionPipeline::reconstruct() {
    if (images.size() < 2) { qWarning() << "Need at least 2 images!"; return false; }
    points3D.clear(); colors.clear();
    m_usedTrackBasedGroundTruth = false;

    QString cachePath;
    if (!imageFiles.empty()) {
        cachePath = QFileInfo(imageFiles[0]).absolutePath() + "/" + kReconstructionCacheFileName;
    }

    // ---- NHÁNH CACHE ----
    if (!cachePath.isEmpty() && QFile::exists(cachePath)) {
        qDebug() << "Reconstruction: Found cache at" << cachePath;
        PointCloudT::Ptr cloud(new PointCloudT);
        if (pcl::io::loadPLYFile<PointT>(cachePath.toStdString(), *cloud)!= -1 &&!cloud->empty()) {
            points3D.reserve(cloud->size());
            colors.reserve(cloud->size());
            for (const auto &p : cloud->points) {
                points3D.emplace_back(p.x, p.y, p.z);
                colors.emplace_back(p.b, p.g, p.r);
            }
            qDebug() << "Reconstruction: Loaded" << points3D.size() << "pts from cache.";

            // Texture lại bằng best-view
            textureFromImages(points3D, colors, images, camParams);

            // Lưu lại cache đã texture best-view
            {
                PointCloudT::Ptr textured(new PointCloudT);
                textured->resize(points3D.size());
                for (size_t i=0;i<points3D.size();++i){
                    auto &p = textured->points[i];
                    p.x=points3D[i].x; p.y=points3D[i].y; p.z=points3D[i].z;
                    p.b=colors[i][0]; p.g=colors[i][1]; p.r=colors[i][2];
                }
                pcl::io::savePLYFileBinary(cachePath.toStdString(), *textured);
                qDebug() << "Reconstruction: Saved TEXTURED cache to" << cachePath;
            }

            // Tạo lại mesh riêng để xuất file, không dùng làm cache
            auto mesh = poissonMeshing(points3D, colors, 10);
            textureMeshFromImages(mesh, images, camParams);
            return!points3D.empty();
        }
    }

    // ---- TÍNH LẠI TỪ ĐẦU ----
    keypoints.resize(images.size());
    descriptors.resize(images.size());
    qDebug() << "Extracting features from" << images.size() << "images in parallel...";
    QElapsedTimer timer; timer.start();
    QVector<int> indices(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    QtConcurrent::blockingMap(indices, [this](int idx) { extractFeatures(idx); });
    qDebug() << "Feature extraction completed in" << timer.elapsed() << "ms";
    for (size_t i = 0; i < images.size(); ++i)
        qDebug() << " Image" << i << ":" << keypoints[i].size() << "pts";

    bool ok = false;
    if (hasGroundTruthParams && (int)camParams.size() >= (int)images.size())
        ok = reconstructWithGroundTruth();
    else
        ok = reconstructWithEstimatedPose();

    if (!ok || points3D.empty()) { qWarning() << "Reconstruction failed."; return false; }

    qDebug() << "Raw points:" << points3D.size();
    processPointCloud();
    // filterFarOutliers không cần nữa vì vòm sinh ra từ Poisson, đã trim trong poissonMeshing
    densifyPointCloudMLS(points3D, colors, 1000000);

    // 1. Texture point cloud 988k bằng best-view
    textureFromImages(points3D, colors, images, camParams);

    // 2. Lưu cache 988k cho Qt - KHÔNG lấy mesh 3.2M làm cache
    if (!cachePath.isEmpty() &&!points3D.empty()) {
        PointCloudT::Ptr cloud(new PointCloudT);
        cloud->resize(points3D.size());
        for (size_t i=0;i<points3D.size();++i){
            auto &p = cloud->points[i];
            p.x=points3D[i].x; p.y=points3D[i].y; p.z=points3D[i].z;
            p.b=colors[i][0]; p.g=colors[i][1]; p.r=colors[i][2];
        }
        pcl::io::savePLYFileBinary(cachePath.toStdString(), *cloud);
        qDebug() << "Reconstruction: Saved 988k TEXTURED cache to" << cachePath;
    }

    // 3. Poisson + texture mesh -> ra file temple_final_textured_2M.ply riêng (đã trim vòm)
    auto mesh = poissonMeshing(points3D, colors, 10);
    textureMeshFromImages(mesh, images, camParams);

    return!points3D.empty();
}

// ─── Accessors ───────────────────────────────────────────────────────────────

std::vector<cv::Point3f> ReconstructionPipeline::getPointCloud()  const { return points3D; }
std::vector<cv::Vec3b>   ReconstructionPipeline::getPointColors() const { return colors; }
