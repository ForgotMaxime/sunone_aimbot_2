#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <Windows.h>
#include "imgui/imgui.h"
#include "sunone_aimbot_2.h"
#include "overlay/draw_settings.h"
#include "overlay/overlay.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"
#include "nodes/node_graph.h"
#include "nodes/vision_outputs.h"
#include "runtime/thread_loops.h"
#include "scr/data_collector.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <vector>

namespace
{
double FrameInferenceMs()
{
#ifdef USE_CUDA
    return trt_detector.lastInferenceTime.count();
#else
    if (dml_detector)
        return dml_detector->lastInferenceTimeDML.count();
    return 0.0;
#endif
}

ImVec4 ConfidenceColor(double confidence)
{
    const float t = std::clamp(static_cast<float>(confidence), 0.0f, 1.0f);
    return ImVec4(1.0f - t, 0.25f + 0.75f * t, 0.25f, 1.0f); // red -> green
}

float BoxOverlapRatio(const cv::Rect& a, const cv::Rect& b)
{
    const int x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1)
        return 0.0f;
    const float denom = static_cast<float>(std::min(
        static_cast<long long>(a.width) * a.height,
        static_cast<long long>(b.width) * b.height));
    if (denom <= 0.0f)
        return 0.0f;
    return static_cast<float>(x2 - x1) * static_cast<float>(y2 - y1) / denom;
}

// Best matching detection confidence for a tracker box (same space, same
// overlap rule as the game overlay renderer).
double MatchConfidence(const TrackDebugInfo& track,
    const std::vector<NodeGraph::Detection>& detections)
{
    double best = -1.0;
    float bestOverlap = 0.35f;
    const cv::Rect box(track.box);
    for (const auto& detection : detections)
    {
        if (detection.classId != track.classId)
            continue;
        const cv::Rect candidate(static_cast<int>(detection.x), static_cast<int>(detection.y),
            static_cast<int>(detection.width), static_cast<int>(detection.height));
        const float overlap = BoxOverlapRatio(box, candidate);
        if (overlap > bestOverlap)
        {
            bestOverlap = overlap;
            best = detection.confidence;
        }
    }
    return best;
}
}

void draw_debug()
{
    static NodeGraph::Workspace workspace;static NodeGraph::Telemetry telemetry;
    static std::vector<TrackDebugInfo> tracks;static int locked=-1;static double sampled=-1;
    static std::vector<std::shared_ptr<const NodeVision::OverlayPacket>> packets;
    static double inferenceMs = 0.0;
    if(ImGui::GetTime()-sampled>=0.15||sampled<0)
    {
        workspace=NodeGraph::service().workspace();telemetry=NodeGraph::service().telemetry();
        {std::lock_guard<std::mutex> lock(g_trackerDebugMutex);tracks=g_trackerDebugTracks;locked=g_trackerLockedId;}
        packets=NodeVision::overlayPackets();
        inferenceMs=FrameInferenceMs();
        sampled=ImGui::GetTime();
    }
    if(OverlayUI::BeginSection("Diagnostics","diagnostic_options"))
    {
        if(OverlayUI::CheckboxRow("Verbose console output",&config.verbose))OverlayConfig_MarkDirty();
        if(ImGui::Button("OpenCV build information"))std::cout<<cv::getBuildInformation()<<std::endl;
        ImGui::PushTextWrapPos(0);ImGui::TextDisabled("Capture, recording and screenshot settings are edited in their workflow nodes.");ImGui::PopTextWrapPos();
        OverlayUI::EndSection();
    }
    if(OverlayUI::BeginSection("Active workflow nodes","diagnostic_nodes"))
    {
        bool any=false;
        for(std::size_t graph=0;graph<workspace.graphs.size();++graph)
        {
            const auto& page=workspace.graphs[graph];if(!page.active||!page.enabled)continue;
            for(const auto& node:page.nodes)
            {
                using K=NodeGraph::NodeKind;
                if(node.kind!=K::CaptureSource&&node.kind!=K::Inference&&node.kind!=K::DepthInference&&node.kind!=K::DatasetOutput&&node.kind!=K::CollectDataset&&node.kind!=K::ScreenshotOutput&&node.kind!=K::OverlayOutput)continue;
                any=true;ImGui::PushID(static_cast<int>(graph));ImGui::PushID(node.id);
                ImGui::Separator();ImGui::Text("%s / %s",NodeGraph::workflowTitle(page).c_str(),NodeGraph::pageTitle(page).c_str());
                ImGui::Text("%s #%d",NodeGraph::definition(node.kind).title,node.id);
                const NodeGraph::NodeValue* value=nullptr;
                if(telemetry.graphRevisions[graph]==NodeGraph::service().revision())
                    for(const auto& item:telemetry.graphs[graph])if(item.nodeId==node.id){value=&item;break;}
                if(node.disabled)ImGui::TextDisabled("Disabled");else if(node.bypassed)ImGui::TextDisabled("Bypassed");
                else if(value&&!value->value.error.empty()){ImGui::PushTextWrapPos(0);ImGui::TextColored(ImVec4(1,0.4f,0.4f,1),"%s",value->value.error.c_str());ImGui::PopTextWrapPos();}
                else ImGui::TextDisabled("%s",value&&value->active?"Active":"Waiting");
                if(node.kind==K::DatasetOutput||node.kind==K::CollectDataset)
                {
                    const auto state=cvm::GetDataCollectionNodeUiState(static_cast<int>(graph)*NodeGraph::NodeKeyStride+node.id);
                    ImGui::Text("Images: %llu   Labels: %llu",static_cast<unsigned long long>(state.saved_image_count),static_cast<unsigned long long>(state.saved_label_count));
                    if(!state.resolved_output_dir.empty()){ImGui::PushTextWrapPos(0);ImGui::TextUnformatted(state.resolved_output_dir.c_str());ImGui::PopTextWrapPos();}
                    if(!state.status.empty()){ImGui::PushTextWrapPos(0);ImGui::TextUnformatted(state.status.c_str());ImGui::PopTextWrapPos();}
                }
                if(ImGui::Button("Open node"))Overlay_OpenGraphNode(static_cast<int>(graph),node.id);
                ImGui::PopID();ImGui::PopID();
            }
        }
        if(!any)ImGui::TextDisabled("No source or output nodes in active pages.");
        OverlayUI::EndSection();
    }
    if(OverlayUI::BeginSection("Tracker","diagnostic_tracker"))
    {
        ImGui::Text("Latest tracker: %zu tracks / locked ID %d / inference %.2f ms",tracks.size(),locked,inferenceMs);
        ImGui::PushTextWrapPos(0);ImGui::TextDisabled("Box color follows detection confidence (red low, green high). Inference ms is the last per-frame detector timing.");ImGui::PopTextWrapPos();
        std::vector<NodeGraph::Detection> detections;
        if(!packets.empty()&&packets.front()&&packets.front()->detections)detections=packets.front()->detections->objects;
        if(ImGui::BeginTable("##tracks",9,ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_ScrollY,ImVec2(0,240)))
        {
            for(const char* name:{"ID","Class","Conf","Locked","Observed","Missed","Position","Speed","Infer ms"})ImGui::TableSetupColumn(name);
            ImGui::TableHeadersRow();ImGuiListClipper clipper;clipper.Begin(static_cast<int>(tracks.size()));
            while(clipper.Step())for(int i=clipper.DisplayStart;i<clipper.DisplayEnd;++i)
            {
                const auto& track=tracks[i];ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::Text("%d",track.trackId);
                ImGui::TableNextColumn();ImGui::Text("%d",track.classId);ImGui::TableNextColumn();
                const double confidence=MatchConfidence(track,detections);
                if(confidence>=0.0)ImGui::TextColored(ConfidenceColor(confidence),"%.0f%%",confidence*100.0);
                else ImGui::TextDisabled("--");
                ImGui::TableNextColumn();ImGui::TextUnformatted(track.isLocked?"Yes":"No");
                ImGui::TableNextColumn();ImGui::TextUnformatted(track.observedThisFrame?"Yes":"No");ImGui::TableNextColumn();ImGui::Text("%d",track.missedFrames);
                ImGui::TableNextColumn();ImGui::Text("%.0f, %.0f",track.pivotX,track.pivotY);ImGui::TableNextColumn();ImGui::Text("%.0f",std::hypot(track.velocityX,track.velocityY));
                ImGui::TableNextColumn();ImGui::Text("%.2f",inferenceMs);
            }
            ImGui::EndTable();
        }
        OverlayUI::EndSection();
    }
    if(OverlayUI::BeginSection("FOV ring","diagnostic_fov"))
    {
        if(!config.circle_fov_enabled)ImGui::TextDisabled("Circle FOV is off (circle_fov_enabled=false).");
        else ImGui::Text("Circle FOV radius: %d%% of the capture area",config.circle_fov_radius_percent);
        const float size=std::clamp(std::min(ImGui::GetContentRegionAvail().x,220.0f),40.0f,220.0f);
        ImGui::InvisibleButton("##fov_canvas",ImVec2(size,size));
        ImDrawList* preview=ImGui::GetWindowDrawList();
        const ImVec2 min=ImGui::GetItemRectMin(),max=ImGui::GetItemRectMax();
        const ImVec2 center((min.x+max.x)*0.5f,(min.y+max.y)*0.5f);
        const float radius=size*0.5f*std::clamp(config.circle_fov_radius_percent,1,100)/100.0f;
        preview->AddRect(min,max,IM_COL32(120,120,120,255),2.0f,0,1.0f);
        preview->AddLine(ImVec2(center.x-6,center.y),ImVec2(center.x+6,center.y),IM_COL32(255,255,255,220),1.5f);
        preview->AddLine(ImVec2(center.x,center.y-6),ImVec2(center.x,center.y+6),IM_COL32(255,255,255,220),1.5f);
        preview->AddCircle(center,std::max(1.0f,radius),config.circle_fov_enabled?IM_COL32(0,255,0,230):IM_COL32(120,120,120,160),64,2.0f);
        OverlayUI::EndSection();
    }
}
