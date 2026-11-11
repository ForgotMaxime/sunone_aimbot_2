#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <algorithm>

#include "imgui/imgui.h"
#include "sunone_aimbot_2.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

void draw_overlay()
{
    constexpr int kMinReadableOpacity = 252;

    if (OverlayUI::BeginSection("Visual", "overlay_section_visual"))
    {
        {
            const auto row = OverlayUI::BeginSettingRow("Overlay Opacity");
            int prev_opacity = config.overlay_opacity;
            if (ImGui::SliderInt("##overlay_opacity_slider", &config.overlay_opacity, kMinReadableOpacity, 255))
            {
                if (config.overlay_opacity < kMinReadableOpacity) config.overlay_opacity = kMinReadableOpacity;
                if (config.overlay_opacity > 255) config.overlay_opacity = 255;

                Overlay_SetOpacity(config.overlay_opacity);

                if (config.overlay_opacity != prev_opacity)
                    OverlayConfig_MarkDirty();
            }
            OverlayUI::EndSettingRow(row);
        }

        float ui_scale = std::clamp(config.overlay_ui_scale, 0.85f, 1.35f);
        if (ui_scale != config.overlay_ui_scale)
            config.overlay_ui_scale = ui_scale;

        {
            const auto row = OverlayUI::BeginSettingRow("UI Fine Scale");
            if (ImGui::SliderFloat("##overlay_ui_scale_slider", &ui_scale, 0.85f, 1.35f, "%.2f"))
            {
                config.overlay_ui_scale = ui_scale;
                OverlayConfig_MarkDirty();
            }
            OverlayUI::EndSettingRow(row);
        }

        {
            const auto row = OverlayUI::BeginSettingRow("Hide Overlays From Recording");
            if (ImGui::Checkbox("##hide_overlay_from_recording", &config.overlay_exclude_from_capture))
            {
                Overlay_ApplyCaptureExclusion();
                OverlayConfig_MarkDirty();
            }
            OverlayUI::EndSettingRow(row);
        }

        {
            const auto row = OverlayUI::BeginSettingRow("Editor VSync");
            if (ImGui::Checkbox("##overlay_vsync", &config.overlay_vsync))
                OverlayConfig_MarkDirty();
            OverlayUI::EndSettingRow(row);
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("VSync caps the editor at the display rate. Off (default) keeps the lowest input latency.");
        ImGui::PopTextWrapPos();

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Game Overlay", "overlay_section_game"))
    {
        {
            const auto row = OverlayUI::BeginSettingRow("In-Game Overlay");
            if (ImGui::Checkbox("##game_overlay_enabled", &config.game_overlay_enabled))
                OverlayConfig_MarkDirty();
            OverlayUI::EndSettingRow(row);
        }
        {
            int maxFps = std::clamp(config.game_overlay_max_fps, 0, 1000);
            if (maxFps != config.game_overlay_max_fps)
            {
                config.game_overlay_max_fps = maxFps;
                OverlayConfig_MarkDirty();
            }
            const auto row = OverlayUI::BeginSettingRow("Game Overlay Max FPS (0 = uncapped)");
            if (ImGui::SliderInt("##game_overlay_max_fps", &maxFps, 0, 240))
            {
                config.game_overlay_max_fps = std::clamp(maxFps, 0, 1000);
                OverlayConfig_MarkDirty();
            }
            OverlayUI::EndSettingRow(row);
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Box colors, thickness and FOV ring are edited in the graph OverlayOutput node.");
        ImGui::PopTextWrapPos();

        OverlayUI::EndSection();
    }
}
