#include "overlay/config_dirty.h"

#include <cstring>

#include "config.h"
#include "imgui/imgui.h"

extern Config config;

namespace
{
    bool cfgDirty = false;
    double cfgDirtyAt = 0.0;
    constexpr double kSaveDelaySec = 0.35;

    double cfgSavedAt = -100.0;
    double cfgReloadedAt = -100.0;
    char cfgSavedName[64] = "config.ini";
    constexpr double kSaveToastSec = 2.0;
    constexpr double kReloadToastSec = 2.5;
}

void OverlayConfig_MarkDirty()
{
    cfgDirty = true;
    cfgDirtyAt = ImGui::GetTime();
}

bool OverlayConfig_IsDirty()
{
    return cfgDirty;
}

void OverlayConfig_NotifySaved(const char* filename)
{
    cfgSavedAt = ImGui::GetTime();
    const char* name = filename ? filename : "config.ini";
    ::strncpy_s(cfgSavedName, name, sizeof(cfgSavedName) - 1);
    cfgSavedName[sizeof(cfgSavedName) - 1] = '\0';
}

void OverlayConfig_NotifyReloaded()
{
    cfgReloadedAt = ImGui::GetTime();
}

void OverlayConfig_TrySave(const char* filename)
{
    if (!cfgDirty)
        return;

    const double now = ImGui::GetTime();
    if ((now - cfgDirtyAt) < kSaveDelaySec)
        return;

    if (ImGui::IsAnyItemActive())
        return;

    config.saveConfig(filename ? filename : "config.ini");
    cfgDirty = false;
    OverlayConfig_NotifySaved(filename);
}

void OverlayConfig_SaveNow(const char* filename)
{
    if (!cfgDirty)
        return;

    config.saveConfig(filename ? filename : "config.ini");
    cfgDirty = false;
    OverlayConfig_NotifySaved(filename);
}

void OverlayConfig_DrawStatusToast()
{
    const double now = ImGui::GetTime();
    const char* message = nullptr;
    ImVec4 color(0.55f, 1.0f, 0.6f, 1.0f);
    if ((now - cfgReloadedAt) < kReloadToastSec)
    {
        message = "Settings reloaded (F4)";
        color = ImVec4(0.55f, 0.85f, 1.0f, 1.0f);
    }
    else if ((now - cfgSavedAt) < kSaveToastSec)
    {
        message = "Saved";
    }
    else
    {
        return;
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(viewport->Pos.x + viewport->Size.x - 14.0f,
               viewport->Pos.y + viewport->Size.y - 14.0f),
        ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.88f);
    if (ImGui::Begin("##cfg_status_toast", nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_AlwaysAutoResize))
    {
        if ((now - cfgReloadedAt) < kReloadToastSec)
            ImGui::TextColored(color, "%s", message);
        else
            ImGui::TextColored(color, "%s %s", message, cfgSavedName);
        if (cfgDirty)
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1.0f), "%s", "(edited...)");
        }
    }
    ImGui::End();
}
