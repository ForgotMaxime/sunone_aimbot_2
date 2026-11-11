#pragma once
// Lightweight motion-only tracker (ByteTrack spirit, header-only).
// Used on the vision_runtime side: hysteresis 0.35/0.15 + EMA centre/taille
// + constant-velocity extrapolation driven by frameTimestamp.
// No appearance embedding: IoU greedy matching only, O(N*M) with N,M <= 50.
//
// Usage:
//   MotionTracker tracker;
//   tracker.update(boxes, classes, confidences, timestampSeconds, frameW, frameH);
//   -> smoothed boxes/classes/confidences, tracks every frame even when the
//      detector is skipped (detect 1/2 frames) or returns empty.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

struct MotionTracker
{
    struct Item
    {
        cv::Rect box;
        int classId = -1;
        float confidence = 0.0f;
        int trackId = -1;
    };

    // Hysteresis thresholds: confirm >= 0.35, drop below 0.15.
    float confirmThreshold = 0.35f;
    float dropThreshold = 0.15f;
    // EMA smoothing for centre/size.
    float emaAlpha = 0.45f;
    // IoU match threshold + frames a confirmed track survives without match.
    float iouThreshold = 0.30f;
    int maxMissedFrames = 5;
    // Velocity clamp (pixels/second) to avoid explosions on teleport.
    float maxVelocity = 20000.0f;

    std::vector<Item> update(const std::vector<cv::Rect>& boxes,
        const std::vector<int>& classes,
        const std::vector<float>& confidences,
        double timestampSeconds,
        int frameWidth,
        int frameHeight)
    {
        struct Raw { float cx, cy, w, h; int cls; float conf; };
        std::vector<Raw> raw;
        raw.reserve(boxes.size());
        for (std::size_t i = 0; i < boxes.size(); ++i)
        {
            const auto& b = boxes[i];
            if (b.width <= 0 || b.height <= 0) continue;
            const float conf = i < confidences.size() ? confidences[i] : 0.0f;
            if (conf < dropThreshold) continue; // hysteresis low gate
            raw.push_back({b.x + b.width * 0.5f, b.y + b.height * 0.5f,
                static_cast<float>(b.width), static_cast<float>(b.height),
                i < classes.size() ? classes[i] : -1, conf});
        }

        double dt = 0.0;
        if (hasTime && timestampSeconds > lastTimestamp)
            dt = std::min(timestampSeconds - lastTimestamp, 0.5);
        lastTimestamp = timestampSeconds;
        hasTime = true;

        // Predict existing tracks forward (constant velocity).
        for (auto& t : tracks)
        {
            t.px += t.vx * static_cast<float>(dt);
            t.py += t.vy * static_cast<float>(dt);
        }

        std::vector<bool> used(raw.size(), false);
        // Greedy IoU matching, best confidence first.
        std::vector<std::size_t> order(raw.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return raw[a].conf > raw[b].conf; });

        for (auto& t : tracks) t.matched = false;

        for (std::size_t oi : order)
        {
            const auto& r = raw[oi];
            // Predicted box of each track.
            float bestIou = iouThreshold;
            int best = -1;
            const float rx1 = r.cx - r.w * 0.5f, ry1 = r.cy - r.h * 0.5f;
            const float rx2 = r.cx + r.w * 0.5f, ry2 = r.cy + r.h * 0.5f;
            for (std::size_t ti = 0; ti < tracks.size(); ++ti)
            {
                auto& t = tracks[ti];
                if (t.matched) continue;
                if (t.classId != -1 && r.cls != -1 && t.classId != r.cls) continue;
                const float tx1 = t.px - t.pw * 0.5f, ty1 = t.py - t.ph * 0.5f;
                const float tx2 = t.px + t.pw * 0.5f, ty2 = t.py + t.ph * 0.5f;
                const float ix1 = std::max(rx1, tx1), iy1 = std::max(ry1, ty1);
                const float ix2 = std::min(rx2, tx2), iy2 = std::min(ry2, ty2);
                const float iw = ix2 - ix1, ih = iy2 - iy1;
                if (iw <= 0 || ih <= 0) continue;
                const float inter = iw * ih;
                const float uni = r.w * r.h + t.pw * t.ph - inter;
                if (uni <= 0) continue;
                const float iou = inter / uni;
                if (iou > bestIou) { bestIou = iou; best = static_cast<int>(ti); }
            }
            if (best < 0) continue;
            used[oi] = true;
            auto& t = tracks[best];
            t.matched = true;
            t.missed = 0;
            ++t.hits;
            // EMA update.
            const float a = emaAlpha;
            const float ncx = a * r.cx + (1 - a) * t.px;
            const float ncy = a * r.cy + (1 - a) * t.py;
            const float nw = a * r.w + (1 - a) * t.pw;
            const float nh = a * r.h + (1 - a) * t.ph;
            if (dt > 1e-6)
            {
                float vx = (ncx - t.px) / static_cast<float>(dt);
                float vy = (ncy - t.py) / static_cast<float>(dt);
                const float v = std::sqrt(vx * vx + vy * vy);
                if (v > maxVelocity) { vx *= maxVelocity / v; vy *= maxVelocity / v; }
                // Light velocity smoothing.
                t.vx = 0.6f * vx + 0.4f * t.vx;
                t.vy = 0.6f * vy + 0.4f * t.vy;
            }
            t.px = ncx; t.py = ncy; t.pw = nw; t.ph = nh;
            t.classId = r.cls;
            t.confidence = std::max(t.confidence, r.conf);
            if (!t.confirmed && (r.conf >= confirmThreshold || t.hits >= 2))
                t.confirmed = true;
        }

        // Unmatched raw -> new tentative tracks (only if above confirm gate
        // or second sight; low gate already applied above).
        for (std::size_t i = 0; i < raw.size(); ++i)
        {
            if (used[i]) continue;
            const auto& r = raw[i];
            Track t;
            t.px = r.cx; t.py = r.cy; t.pw = r.w; t.ph = r.h;
            t.vx = t.vy = 0;
            t.classId = r.cls; t.confidence = r.conf;
            t.hits = 1; t.missed = 0; t.matched = true;
            t.confirmed = (r.conf >= confirmThreshold);
            t.trackId = nextId++;
            tracks.push_back(t);
        }

        // Age unmatched tracks; drop stale or weak ones.
        for (auto& t : tracks)
            if (!t.matched) { ++t.missed; t.confidence *= 0.9f; }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
            [&](const Track& t) {
                if (t.missed > maxMissedFrames) return true;
                if (!t.confirmed && t.missed >= 2) return true;
                if (t.confirmed && t.confidence < dropThreshold && t.missed >= 2) return true;
                return false;
            }), tracks.end());

        // Cap total tracks (max_det friendly).
        if (tracks.size() > 50)
        {
            std::nth_element(tracks.begin(), tracks.begin() + 50, tracks.end(),
                [](const Track& a, const Track& b) { return a.confidence > b.confidence; });
            tracks.resize(50);
        }

        // Emit: confirmed tracks always; tentative only if fresh match.
        std::vector<Item> out;
        out.reserve(tracks.size());
        for (const auto& t : tracks)
        {
            if (!t.confirmed && !t.matched) continue;
            const float w = std::max(1.0f, t.pw), h = std::max(1.0f, t.ph);
            float x = t.px - w * 0.5f, y = t.py - h * 0.5f;
            x = std::clamp(x, 0.0f, static_cast<float>(std::max(0, frameWidth - 1)));
            y = std::clamp(y, 0.0f, static_cast<float>(std::max(0, frameHeight - 1)));
            const float ww = std::clamp(w, 1.0f, static_cast<float>(frameWidth) - x);
            const float hh = std::clamp(h, 1.0f, static_cast<float>(frameHeight) - y);
            if (ww <= 0 || hh <= 0) continue;
            Item item;
            item.box = cv::Rect(static_cast<int>(std::lround(x)),
                static_cast<int>(std::lround(y)),
                std::max(1, static_cast<int>(std::lround(ww))),
                std::max(1, static_cast<int>(std::lround(hh))));
            item.classId = t.classId;
            item.confidence = std::clamp(t.confidence, 0.0f, 1.0f);
            item.trackId = t.trackId;
            out.push_back(item);
        }
        std::sort(out.begin(), out.end(),
            [](const Item& a, const Item& b) { return a.confidence > b.confidence; });
        return out;
    }

    void clear() { tracks.clear(); hasTime = false; lastTimestamp = 0; }

    std::size_t size() const { return tracks.size(); }

private:
    struct Track
    {
        float px = 0, py = 0, pw = 0, ph = 0;
        float vx = 0, vy = 0;
        int classId = -1;
        float confidence = 0;
        int hits = 0, missed = 0;
        bool matched = false, confirmed = false;
        int trackId = -1;
    };
    std::vector<Track> tracks;
    int nextId = 0;
    double lastTimestamp = 0;
    bool hasTime = false;
};
