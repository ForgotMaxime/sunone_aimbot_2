#pragma once
#include "nodes/node_graph.h"
#include <opencv2/core.hpp>
#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#endif
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace NodeVision
{
struct FrameStorage
{
    mutable std::mutex cpuMutex;
    mutable cv::Mat image;
    cv::Mat source;
#ifdef USE_CUDA
    cv::cuda::GpuMat gpu;
#endif
};
struct MaskStorage { cv::Mat mask; };
using Infer = std::function<NodeGraph::DetectionStream(int,const NodeGraph::FrameStream&,const NodeGraph::Node&)>;

// Snapshot provider-owned buffers before they can be reused by capture.
NodeGraph::FrameStream makeFrame(const cv::Mat& image,const cv::Mat& source,
    std::chrono::steady_clock::time_point timestamp,std::uint64_t generation);
#ifdef USE_CUDA
NodeGraph::FrameStream makeFrameGpu(const cv::cuda::GpuMat& image,
    std::chrono::steady_clock::time_point timestamp,std::uint64_t generation);
#endif
std::shared_ptr<const FrameStorage> storage(const NodeGraph::FrameStream& frame);
cv::Mat cpuImage(const NodeGraph::FrameStream& frame);
cv::Mat sourceImage(const NodeGraph::FrameStream& frame);
std::string resolveModelPath(const NodeGraph::Node& node);
void reportError(int nodeKey,const std::string& error);
std::string nodeError(int nodeKey);

// P0 capture-latency: lock the latency-optimal inference flags (CUDA graph +
// static FP16 batch=1). Returns a sanitized copy with the flags forced and,
// when warning is non-null, a description of what was corrected.
NodeGraph::Node sanitizeInferenceNode(const NodeGraph::Node& node, std::string* warning = nullptr);
// Validates the locked configuration: fixed_input_size=1, use_cuda_graph=1,
// use_pinned_memory=1, export_enable_fp16=1, export_enable_fp8=0.
// Returns true when valid; otherwise fills *error.
bool validateStaticFp16Batch1(const NodeGraph::Node& node, std::string* error = nullptr);
// Mirrors the DDA no-present skip (AccumulatedFrames==0 without a present).
bool shouldSkipStaleCapture(std::uint32_t accumulatedFrames, bool hasLastPresentTime);

// Called without configMutex. Applies node settings to the physical capture
// service and returns false when the graph has no active capture source.
bool syncCaptureConfiguration(bool force = false);
NodeGraph::Actions processFrame(const NodeGraph::FrameStream& frame,const Infer& infer);
}
