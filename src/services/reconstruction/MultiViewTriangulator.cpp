#include "MultiViewTriangulator.h"
#include <Eigen/Dense>
#include <cmath>

static cv::Point3d cameraCenter(const CameraParams &cp) {
    cv::Mat C = -cp.R.t() * cp.t;   // C = -R^T * t
    return cv::Point3d(C.at<double>(0), C.at<double>(1), C.at<double>(2));
}

double MultiViewTriangulator::maxPairwiseAngleDeg(const FeatureTrack &track,
                                                  const std::vector<CameraParams> &camParams)
{
    double best = 0.0;
    cv::Point3d X(track.point3D.x, track.point3D.y, track.point3D.z);
    for (size_t i = 0; i < track.observations.size(); ++i) {
        for (size_t j = i + 1; j < track.observations.size(); ++j) {
            cv::Point3d Ci = cameraCenter(camParams[track.observations[i].first]);
            cv::Point3d Cj = cameraCenter(camParams[track.observations[j].first]);
            cv::Point3d di = X - Ci, dj = X - Cj;
            double ni = cv::norm(di), nj = cv::norm(dj);
            if (ni < 1e-9 || nj < 1e-9) continue;
            double cosA = std::clamp(di.dot(dj) / (ni * nj), -1.0, 1.0);
            best = std::max(best, std::acos(cosA) * 180.0 / CV_PI);
        }
    }
    return best;
}

bool MultiViewTriangulator::triangulateTrack(FeatureTrack &track,
                                             const std::vector<CameraParams> &camParams,
                                             const std::vector<std::vector<cv::KeyPoint>> &kps,
                                             const Params &params)
{
    int n = (int)track.observations.size();
    if (n < 2) return false;

    Eigen::MatrixXd A(2 * n, 4);
    for (int i = 0; i < n; ++i) {
        int imgIdx = track.observations[i].first;
        int kpIdx  = track.observations[i].second;
        const cv::Mat &Pm = camParams[imgIdx].P;
        const cv::Point2f &pt = kps[imgIdx][kpIdx].pt;
        for (int c = 0; c < 4; ++c) {
            A(2*i,   c) = pt.x * Pm.at<double>(2,c) - Pm.at<double>(0,c);
            A(2*i+1, c) = pt.y * Pm.at<double>(2,c) - Pm.at<double>(1,c);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    if (std::abs(X(3)) < 1e-9) return false;
    X /= X(3);
    cv::Point3f pt3d((float)X(0), (float)X(1), (float)X(2));

    std::vector<std::pair<int,int>> keptObs;
    for (auto &obs : track.observations) {
        int imgIdx = obs.first, kpIdx = obs.second;
        const cv::Mat &Pm = camParams[imgIdx].P;
        cv::Mat p4 = (cv::Mat_<double>(4,1) << pt3d.x, pt3d.y, pt3d.z, 1.0);
        cv::Mat proj = Pm * p4;
        double z = proj.at<double>(2);
        if (z <= 0) continue;
        double u = proj.at<double>(0) / z, v = proj.at<double>(1) / z;
        const cv::Point2f &obsPt = kps[imgIdx][kpIdx].pt;
        double err = std::sqrt((u-obsPt.x)*(u-obsPt.x) + (v-obsPt.y)*(v-obsPt.y));
        if (err <= params.maxReprojError) keptObs.push_back(obs);
    }

    if ((int)keptObs.size() < std::min(2, params.minObservations)) return false;
    track.observations = keptObs;
    track.point3D = pt3d;

    // Guard: 2-view với baseline quá hẹp → nghiệm SVD không ổn định, dễ ra điểm sai xa
    if (track.observations.size() == 2) {
        if (maxPairwiseAngleDeg(track, camParams) < params.minTriangulationAngleDeg)
            return false;
    }

    track.valid = true;
    return true;
}

void MultiViewTriangulator::refinePoint(FeatureTrack &track,
                                        const std::vector<CameraParams> &camParams,
                                        const std::vector<std::vector<cv::KeyPoint>> &kps,
                                        int iterations)
{
    Eigen::Vector3d X(track.point3D.x, track.point3D.y, track.point3D.z);
    for (int it = 0; it < iterations; ++it) {
        Eigen::MatrixXd J(2 * track.observations.size(), 3);
        Eigen::VectorXd r(2 * track.observations.size());
        for (size_t i = 0; i < track.observations.size(); ++i) {
            int imgIdx = track.observations[i].first;
            int kpIdx  = track.observations[i].second;
            const cv::Mat &Pm = camParams[imgIdx].P;
            const cv::Point2f &obsPt = kps[imgIdx][kpIdx].pt;

            double px = Pm.at<double>(0,0)*X(0)+Pm.at<double>(0,1)*X(1)+Pm.at<double>(0,2)*X(2)+Pm.at<double>(0,3);
            double py = Pm.at<double>(1,0)*X(0)+Pm.at<double>(1,1)*X(1)+Pm.at<double>(1,2)*X(2)+Pm.at<double>(1,3);
            double pz = Pm.at<double>(2,0)*X(0)+Pm.at<double>(2,1)*X(1)+Pm.at<double>(2,2)*X(2)+Pm.at<double>(2,3);
            if (std::abs(pz) < 1e-9) pz = 1e-9;

            double u = px / pz, v = py / pz;
            r(2*i) = u - obsPt.x; r(2*i+1) = v - obsPt.y;
            for (int c = 0; c < 3; ++c) {
                double dpx=Pm.at<double>(0,c), dpy=Pm.at<double>(1,c), dpz=Pm.at<double>(2,c);
                J(2*i,   c) = (dpx*pz - px*dpz)/(pz*pz);
                J(2*i+1, c) = (dpy*pz - py*dpz)/(pz*pz);
            }
        }
        Eigen::Vector3d delta = (J.transpose()*J).ldlt().solve(-J.transpose()*r);
        if (!delta.allFinite()) break;
        X += delta;
        if (delta.norm() < 1e-6) break;
    }
    track.point3D = cv::Point3f((float)X(0), (float)X(1), (float)X(2));
}
