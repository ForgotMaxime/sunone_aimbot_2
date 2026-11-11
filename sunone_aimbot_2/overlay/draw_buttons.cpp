#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "imgui/imgui.h"
#include "sunone_aimbot_2.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"
#include "overlay/preset_store.h"
#include "keycodes.h"

namespace
{
// Key-capture state: only one slot captures at a time. uid identifies the
// (row, slot) pair across all binding rows drawn this frame.
int g_capturingUid = -1;
bool g_captureWaitRelease = false;
double g_captureStartedAt = 0.0;
constexpr double kCaptureTimeoutSec = 10.0;

int findKeyIndexByName(const std::string& keyName)
{
    for (size_t k = 0; k < key_names.size(); ++k)
    {
        if (key_names[k] == keyName)
            return static_cast<int>(k);
    }
    return 0;
}

// Poll the full KeyCodes table through Win32 so mouse buttons and keyboard
// use the same source of truth as the runtime hotkey listener.
std::string PollAnyPressedKey()
{
    for (const auto& pair : KeyCodes::key_code_map)
    {
        if (pair.first == "None")
            continue;
        const int vk = pair.second;
        if (vk <= 0)
            continue;
        if ((GetAsyncKeyState(vk) & 0x8000) != 0)
            return pair.first;
    }
    return {};
}

bool drawKeySlot(const char* label, std::string& keyName, int uid, float width)
{
    bool changed = false;
    const bool capturing = (g_capturingUid == uid);

    ImGui::PushID(uid);
    if (capturing)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.47f, 0.83f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.55f, 0.95f, 1.0f));
    }
    if (ImGui::Button(capturing ? "..." : keyName.c_str(), ImVec2(width, 0.0f)))
    {
        if (!capturing)
        {
            g_capturingUid = uid;
            g_captureWaitRelease = true;
            g_captureStartedAt = ImGui::GetTime();
        }
    }
    if (capturing)
    {
        ImGui::PopStyleColor(2);
        if (ImGui::IsKeyPressed(ImGuiKey_Escape))
        {
            g_capturingUid = -1; // Esc cancels; pick Escape via the fallback list.
        }
        else if (g_captureWaitRelease)
        {
            // Ignore the click that armed the capture.
            if (PollAnyPressedKey().empty())
                g_captureWaitRelease = false;
        }
        else
        {
            const std::string hit = PollAnyPressedKey();
            if (!hit.empty())
            {
                keyName = hit;
                changed = true;
                g_capturingUid = -1;
            }
            else if (ImGui::GetTime() - g_captureStartedAt > kCaptureTimeoutSec)
            {
                g_capturingUid = -1;
            }
        }
    }
    if (!capturing)
    {
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s\nLeft-click: capture a key/button\nRight-click: set to None\n(Esc cancels capture)", label);
        if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && keyName != "None")
        {
            keyName = "None";
            changed = true;
        }
    }
    ImGui::PopID();
    return changed;
}

bool drawButtonBindingRows(const char* rowLabel, std::vector<std::string>& bindings, bool keepAtLeastOne, int& uid)
{
    bool changed = false;
    if (bindings.empty() && keepAtLeastOne)
    {
        bindings.push_back("None");
        changed = true;
    }

    for (size_t i = 0; i < bindings.size();)
    {
        std::string& currentKeyName = bindings[i];
        const std::string indexedLabel = (bindings.size() > 1)
            ? std::string(rowLabel) + " " + std::to_string(i + 1)
            : std::string(rowLabel);

        ImGui::PushID(static_cast<int>(i));

        const auto row = OverlayUI::BeginSettingRow(indexedLabel.c_str());
        const float actionBtnW = ImGui::GetFrameHeight();
        float slotWidth = row.controlWidth - (actionBtnW * 2.0f + 7.0f);
        if (slotWidth < 1.0f)
            slotWidth = 1.0f;

        if (!key_names_cstrs.empty())
        {
            // Primary UI: press-to-capture button.
            if (drawKeySlot(indexedLabel.c_str(), currentKeyName, uid++, slotWidth))
                changed = true;
        }
        else
        {
            // Fallback when the key list is unavailable: classic static combo.
            int currentIndex = findKeyIndexByName(currentKeyName);
            ImGui::SetNextItemWidth(slotWidth);
            if (ImGui::Combo("##value", &currentIndex, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
            {
                currentKeyName = key_names[currentIndex];
                changed = true;
            }
            ++uid;
        }

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("+", ImVec2(actionBtnW, 0.0f)))
        {
            bindings.insert(bindings.begin() + static_cast<std::vector<std::string>::difference_type>(i + 1), "None");
            changed = true;
        }

        ImGui::SameLine(0.0f, 3.0f);
        bool removedCurrent = false;
        if (ImGui::Button("-", ImVec2(actionBtnW, 0.0f)))
        {
            if (bindings.size() <= 1 && keepAtLeastOne)
            {
                bindings[0] = "None";
            }
            else
            {
                bindings.erase(bindings.begin() + static_cast<std::vector<std::string>::difference_type>(i));
                removedCurrent = true;
            }
            changed = true;
        }

        OverlayUI::EndSettingRow(row);
        ImGui::PopID();

        if (removedCurrent)
            continue;

        ++i;
    }

    return changed;
}

void drawBindingRowsAndMarkDirty(const char* rowLabel, std::vector<std::string>& bindings, int& uid, bool keepAtLeastOne = true)
{
    if (drawButtonBindingRows(rowLabel, bindings, keepAtLeastOne, uid))
        OverlayConfig_MarkDirty();
}

void drawBindingConflicts()
{
    const std::pair<const char*, const std::vector<std::string>*> rows[] = {
        {"Targeting", &config.button_targeting},
        {"Shoot", &config.button_shoot},
        {"Zoom", &config.button_zoom},
        {"Exit", &config.button_exit},
        {"Pause", &config.button_pause},
        {"Reload settings and graphs", &config.button_reload_config},
        {"Open Overlay", &config.button_open_overlay},
        {"Screenshot", &config.screenshot_button},
    };

    std::map<std::string, std::vector<std::string>> owners;
    for (const auto& row : rows)
        for (const auto& key : *row.second)
            if (!key.empty() && key != "None")
                owners[key].push_back(row.first);

    bool anyConflict = false;
    for (const auto& entry : owners)
    {
        std::vector<std::string> unique = entry.second;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        if (unique.size() > 1)
        {
            if (!anyConflict)
            {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", "Hotkey conflicts:");
                ImGui::PopTextWrapPos();
                anyConflict = true;
            }
            std::string line = entry.first + " used by ";
            for (size_t i = 0; i < unique.size(); ++i)
            {
                if (i) line += ", ";
                line += unique[i];
            }
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.4f, 1.0f), "%s", line.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    if (!anyConflict)
        ImGui::TextDisabled("No hotkey conflicts. Use None (right-click a slot) to disable a binding.");
}
}

void draw_buttons()
{
    if (OverlayUI::BeginSection("Hotkeys", "buttons_section_hotkeys"))
    {
        int uid = 0;
        drawBindingRowsAndMarkDirty("Targeting", config.button_targeting, uid);
        drawBindingRowsAndMarkDirty("Shoot", config.button_shoot, uid);
        drawBindingRowsAndMarkDirty("Zoom", config.button_zoom, uid);
        drawBindingRowsAndMarkDirty("Exit", config.button_exit, uid);
        drawBindingRowsAndMarkDirty("Pause", config.button_pause, uid);
        drawBindingRowsAndMarkDirty("Reload settings and graphs", config.button_reload_config, uid);
        drawBindingRowsAndMarkDirty("Open Overlay", config.button_open_overlay, uid);

        const auto row = OverlayUI::BeginSettingRow("Arrow tuning of active pages");
        if (ImGui::Checkbox("##value", &config.enable_arrows_settings))
            OverlayConfig_MarkDirty();
        OverlayUI::EndSettingRow(row);
        drawBindingConflicts();
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Game presets", "buttons_section_presets"))
    {
        static PresetStore store;
        static bool storeReady = false;
        static char newPresetName[65] = "";
        if (!storeReady)
        {
            storeReady = true;
            store.ensureLoaded(config.game_profiles);
            std::snprintf(newPresetName, sizeof(newPresetName), "%s", config.active_game.c_str());
        }

        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Presets are stored in presets/*.json (migrated from [Games]). Applying sets the active game and saves config.ini.");
        ImGui::PopTextWrapPos();

        const auto activeIt = config.game_profiles.find(config.active_game);
        if (activeIt != config.game_profiles.end())
        {
            const auto& gp = activeIt->second;
            ImGui::Text("Active: %s (sens %.3g, yaw %.5g, pitch %.5g%s)",
                gp.name.c_str(), gp.sens, gp.yaw, gp.pitch,
                gp.fovScaled ? ", fov-scaled" : "");
        }
        else
        {
            ImGui::TextDisabled("Active: %s (no matching profile)", config.active_game.c_str());
        }

        if (ImGui::Button("Refresh presets"))
            store.refresh();
        if (!store.lastError.empty())
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", store.lastError.c_str());

        for (const auto& preset : store.presets)
        {
            ImGui::PushID(preset.name.c_str());
            const bool isActive = (preset.name == config.active_game);
            if (isActive)
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 1.0f, 0.6f, 1.0f));
            ImGui::BulletText("%s  (sens %.3g, yaw %.5g%s)",
                preset.name.c_str(), preset.sens, preset.yaw,
                preset.fovScaled ? ", fov-scaled" : "");
            if (isActive)
                ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::SmallButton("Apply"))
            {
                config.active_game = preset.name;
                Config::GameProfile gp;
                gp.name = preset.name;
                gp.sens = preset.sens;
                gp.yaw = preset.yaw;
                gp.pitch = preset.pitch;
                gp.fovScaled = preset.fovScaled;
                gp.baseFOV = preset.baseFOV;
                config.game_profiles[preset.name] = gp;
                OverlayConfig_MarkDirty();
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete"))
            {
                if (store.removePreset(preset.name))
                    OverlayConfig_MarkDirty();
            }
            ImGui::PopID();
        }
        if (store.presets.empty())
            ImGui::TextDisabled("No presets yet.");

        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputTextWithHint("##preset_name", "Preset name (Enter = save current game)", newPresetName, sizeof(newPresetName));
        if (ImGui::Button("Save current game as preset"))
        {
            std::string name = newPresetName;
            if (name.empty())
                name = config.active_game;
            GamePreset preset;
            preset.name = name;
            const auto it = config.game_profiles.find(config.active_game);
            if (it != config.game_profiles.end())
            {
                preset.sens = it->second.sens;
                preset.yaw = it->second.yaw;
                preset.pitch = it->second.pitch;
                preset.fovScaled = it->second.fovScaled;
                preset.baseFOV = it->second.baseFOV;
            }
            if (store.savePreset(preset))
            {
                config.game_profiles[name] = Config::GameProfile{ name, preset.sens, preset.yaw,
                    preset.pitch, preset.fovScaled, preset.baseFOV };
                OverlayConfig_MarkDirty();
            }
        }
        if (!store.lastError.empty())
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", store.lastError.c_str());
        OverlayUI::EndSection();
    }
}
