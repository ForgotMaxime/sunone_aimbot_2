#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <deque>

extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;
extern std::atomic<bool> capture_window_changed;
extern std::deque<cv::Mat> frameQueue;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);
extern std::atomic<int> screenWidth;
extern std::atomic<int> screenHeight;

extern std::atomic<int> captureFrameCount;
extern std::atomic<int> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

extern std::atomic<uint64_t> captureWinrtPollAttemptsTotal;
extern std::atomic<uint64_t> captureWinrtFramesDrainedTotal;
extern std::atomic<uint64_t> captureWinrtFramesReturnedTotal;
extern std::atomic<uint64_t> captureWinrtEmptyPollsTotal;
extern std::atomic<uint64_t> captureWinrtReadbackMicrosTotal;
extern std::atomic<uint64_t> captureWinrtMapMicrosTotal;
extern std::atomic<uint64_t> captureWinrtPixelCopyMicrosTotal;

extern cv::Mat latestFrame;

extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;

// P0 capture-latency: SPSC triple-buffer, latest-only.
// Single producer (capture thread) / single consumer (latest-frame reader).
// Seqlock style: writer publishes an even sequence number when the slot is
// stable; readers retry when the sequence is odd or changes mid-copy.
// The legacy deque<cv::Mat>+mutex API above is kept untouched for
// compatibility; capture.cpp migrates to this ring as a follow-up.
class LatestFrameRing
{
public:
    static constexpr std::size_t kSlots = 3;

    void push(const cv::Mat& frame)
    {
        if (frame.empty())
            return;
        const std::uint64_t begin = seq_.fetch_add(1, std::memory_order_acquire) + 1;
        slots_[begin % kSlots] = frame.clone();
        seq_.store(begin + 1, std::memory_order_release);
    }

    // Copy the newest stable frame. Returns false when empty/torn.
    // Writer publishes even seq S after writing slot (S-1)%kSlots.
    bool latest(cv::Mat& out) const
    {
        for (int attempt = 0; attempt < 4; ++attempt)
        {
            const std::uint64_t before = seq_.load(std::memory_order_acquire);
            if ((before & 1u) != 0u || before == 0u)
                continue;
            const cv::Mat snapshot = slots_[(before - 1u) % kSlots];
            const std::uint64_t after = seq_.load(std::memory_order_acquire);
            if (before != after || snapshot.empty())
                continue;
            out = snapshot.clone();
            return true;
        }
        return false;
    }

    bool empty() const { return seq_.load(std::memory_order_acquire) == 0u; }

    std::size_t depth() const { return empty() ? 0u : 1u; }

    void clear()
    {
        for (auto& slot : slots_)
            slot.release();
        seq_.store(0u, std::memory_order_release);
    }

private:
    std::array<cv::Mat, kSlots> slots_;
    std::atomic<std::uint64_t> seq_{0};
};

inline LatestFrameRing& captureLatestRing()
{
    static LatestFrameRing ring;
    return ring;
}

inline void PublishLatestFrame(const cv::Mat& frame) { captureLatestRing().push(frame); }

inline bool TryGetLatestFrame(cv::Mat& out) { return captureLatestRing().latest(out); }

// P0: skip inference when DDA reports no new present (AccumulatedFrames==0
// without a present: mouse-only or metadata-only event, possibly coalesced
// dirty rects). Callers keep the last good frame alive and skip the backend.
inline bool ShouldSkipDdaInference(std::uint32_t accumulatedFrames, bool hasLastPresentTime)
{
    return accumulatedFrames == 0u && !hasLastPresentTime;
}

class IScreenCapture
{
public:
    virtual ~IScreenCapture() {}
    virtual cv::Mat GetNextFrameCpu() = 0;
    bool GetSourceDimensions(int& width, int& height) const
    {
        width = sourceWidth_;
        height = sourceHeight_;
        return width > 0 && height > 0;
    }

protected:
    void SetSourceDimensions(int width, int height)
    {
        sourceWidth_ = width;
        sourceHeight_ = height;
    }

private:
    int sourceWidth_ = 0;
    int sourceHeight_ = 0;
};

#ifdef USE_CUDA
cv::Mat getCurrentDetectionSuppressionMask();
extern std::atomic<uint64_t> captureGpuAttemptsTotal;
extern std::atomic<uint64_t> captureGpuCapturedTotal;
extern std::atomic<uint64_t> captureGpuTimeoutTotal;
extern std::atomic<uint64_t> captureGpuAccumulatedFramesTotal;
extern std::atomic<uint64_t> captureGpuMissedFramesTotal;
extern std::atomic<uint64_t> captureGpuPresentFramesTotal;
extern std::atomic<uint64_t> captureGpuMouseOnlyEventsTotal;
extern std::atomic<uint64_t> captureGpuMetadataOnlyEventsTotal;
extern std::atomic<uint64_t> captureGpuCoalescedEventsTotal;
extern std::atomic<uint64_t> captureCpuFallbackAttemptsTotal;
extern std::atomic<uint64_t> captureCpuFallbackFramesTotal;
#endif

#endif // CAPTURE_H
