#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

// Một track = tập hợp các quan sát (ảnh, keypoint) của CÙNG một điểm 3D thực
struct FeatureTrack {
    std::vector<std::pair<int,int>> observations; // (imageIdx, keypointIdx)
    cv::Point3f point3D;
    cv::Vec3b   color;
    bool        valid = false;
};

// Union-Find để gộp các match pairwise thành track đa-view
class TrackBuilder {
public:
    void addMatch(int imgA, int kpA, int imgB, int kpB) {
        int a = keyOf(imgA, kpA);
        int b = keyOf(imgB, kpB);
        unite(a, b);
    }

    std::vector<FeatureTrack> buildTracks(int minObservations = 2) {
        std::unordered_map<int, std::vector<std::pair<int,int>>> groups;
        for (auto &kv : m_nodeKey) {
            int root = find(kv.first);
            groups[root].push_back(m_keyToObs[kv.first]);
        }
        std::vector<FeatureTrack> tracks;
        tracks.reserve(groups.size());
        for (auto &g : groups) {
            if ((int)g.second.size() < minObservations) continue;
            // Loại track có 2 quan sát trùng ảnh (matching lỗi/xoắn track)
            std::vector<int> seenImgs;
            bool dup = false;
            for (auto &obs : g.second) {
                if (std::find(seenImgs.begin(), seenImgs.end(), obs.first) != seenImgs.end()) {
                    dup = true; break;
                }
                seenImgs.push_back(obs.first);
            }
            if (dup) continue;
            FeatureTrack t;
            t.observations = g.second;
            tracks.push_back(std::move(t));
        }
        return tracks;
    }

private:
    std::unordered_map<int,int> m_parent;
    std::unordered_map<int, std::pair<int,int>> m_keyToObs;
    std::unordered_map<int,int> m_nodeKey; // dummy presence set

    static int keyOf(int img, int kp) { return img * 100000 + kp; } // đủ cho ảnh <100k keypoints

    int find(int x) {
        if (!m_parent.count(x)) { m_parent[x] = x; return x; }
        if (m_parent[x] != x) m_parent[x] = find(m_parent[x]);
        return m_parent[x];
    }

    void unite(int a, int b) {
        if (!m_parent.count(a)) { m_parent[a] = a; m_nodeKey[a]=1; }
        if (!m_parent.count(b)) { m_parent[b] = b; m_nodeKey[b]=1; }
        int ra = find(a), rb = find(b);
        if (ra != rb) m_parent[ra] = rb;
    }

public:
    void registerObs(int img, int kp) {
        int k = keyOf(img, kp);
        m_keyToObs[k] = {img, kp};
    }
};
