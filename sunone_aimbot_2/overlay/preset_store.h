#pragma once

// Lightweight per-game preset store (task scope: PresetStore + UI + dirty).
// Presets live in presets/<name>.json next to config.ini. On first use the
// legacy [Games] profiles are migrated to JSON files (missing files only).
// Applying a preset sets config.active_game and upserts game_profiles so the
// values survive in config.ini; live graphs pick them up after F4 reload.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct GamePreset
{
    std::string name;
    double sens = 1.0;
    double yaw = 0.022;
    double pitch = 0.022;
    bool fovScaled = false;
    double baseFOV = 0.0;
};

class PresetStore
{
public:
    std::vector<GamePreset> presets;
    std::string lastError;

    bool ensureLoaded(const std::unordered_map<std::string, struct Config::GameProfile>& games)
    {
        if (loaded_)
            return true;
        loaded_ = true;
        std::error_code ec;
        std::filesystem::create_directories("presets", ec);
        importFromGames(games);
        return refresh();
    }

    bool refresh()
    {
        presets.clear();
        lastError.clear();
        std::error_code ec;
        std::filesystem::directory_iterator it("presets", ec), end;
        if (ec)
        {
            lastError = "Cannot read presets/.";
            return false;
        }
        for (; !ec && it != end; it.increment(ec))
        {
            std::error_code st;
            if (!it->is_regular_file(st))
                continue;
            if (it->path().extension() != ".json")
                continue;
            GamePreset preset;
            std::string error;
            if (parseFile(it->path().string(), preset, error))
                presets.push_back(preset);
        }
        std::sort(presets.begin(), presets.end(),
            [](const GamePreset& a, const GamePreset& b) { return a.name < b.name; });
        return true;
    }

    bool savePreset(const GamePreset& preset)
    {
        const std::string file = fileFor(preset.name);
        if (file.empty())
        {
            lastError = "Invalid preset name.";
            return false;
        }
        std::error_code ec;
        std::filesystem::create_directories("presets", ec);
        // Atomic write through a temp file.
        const std::string tmp = file + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
            {
                lastError = "Cannot write " + tmp;
                return false;
            }
            out << toJson(preset);
            out.close();
            if (!out)
            {
                lastError = "Cannot write " + tmp;
                std::filesystem::remove(tmp, ec);
                return false;
            }
        }
#ifdef _WIN32
        if (!MoveFileExA(tmp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        {
            lastError = "Cannot replace " + file;
            return false;
        }
#else
        std::filesystem::rename(tmp, file, ec);
        if (ec)
        {
            lastError = "Cannot replace " + file;
            return false;
        }
#endif
        return refresh();
    }

    bool removePreset(const std::string& name)
    {
        const std::string file = fileFor(name);
        if (file.empty())
            return false;
        std::error_code ec;
        std::filesystem::remove(file, ec);
        return refresh() && !ec;
    }

    template <typename GameMap>
    void importFromGames(const GameMap& games)
    {
        for (const auto& kv : games)
        {
            const std::string file = fileFor(kv.first);
            if (file.empty())
                continue;
            std::error_code ec;
            if (std::filesystem::is_regular_file(file, ec))
                continue; // never overwrite user JSON
            GamePreset preset;
            preset.name = kv.first;
            preset.sens = kv.second.sens;
            preset.yaw = kv.second.yaw;
            preset.pitch = kv.second.pitch;
            preset.fovScaled = kv.second.fovScaled;
            preset.baseFOV = kv.second.baseFOV;
            if (!savePreset(preset))
                break;
        }
    }

    static std::string toJson(const GamePreset& preset)
    {
        std::ostringstream out;
        out << "{\n"
            << "  \"name\": \"" << escape(preset.name) << "\",\n"
            << "  \"sens\": " << preset.sens << ",\n"
            << "  \"yaw\": " << preset.yaw << ",\n"
            << "  \"pitch\": " << preset.pitch << ",\n"
            << "  \"fovScaled\": " << (preset.fovScaled ? "true" : "false") << ",\n"
            << "  \"baseFOV\": " << preset.baseFOV << "\n"
            << "}\n";
        return out.str();
    }

private:
    bool loaded_ = false;

    static std::string sanitize(const std::string& name)
    {
        std::string out;
        for (char c : name)
            out += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == ' ') ? c : '_';
        while (!out.empty() && out.front() == ' ')
            out.erase(out.begin());
        while (!out.empty() && out.back() == ' ')
            out.pop_back();
        return out;
    }

    static std::string fileFor(const std::string& name)
    {
        const std::string clean = sanitize(name);
        if (clean.empty() || clean.size() > 64)
            return {};
        return (std::filesystem::path("presets") / (clean + ".json")).string();
    }

    static std::string escape(const std::string& value)
    {
        std::string out;
        for (char c : value)
        {
            if (c == '"' || c == '\\')
                out += '\\';
            out += c;
        }
        return out;
    }

    static bool parseFile(const std::string& path, GamePreset& preset, std::string& error)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            error = "Cannot open " + path;
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();
        auto number = [&](const char* key, double fallback) {
            const size_t at = text.find(key);
            if (at == std::string::npos)
                return fallback;
            const size_t colon = text.find(':', at);
            if (colon == std::string::npos)
                return fallback;
            try
            {
                return std::stod(text.substr(colon + 1));
            }
            catch (...)
            {
                return fallback;
            }
        };
        auto boolean = [&](const char* key, bool fallback) {
            const size_t at = text.find(key);
            if (at == std::string::npos)
                return fallback;
            const size_t colon = text.find(':', at);
            if (colon == std::string::npos)
                return fallback;
            const std::string tail = text.substr(colon + 1, 8);
            if (tail.find("true") != std::string::npos)
                return true;
            if (tail.find("false") != std::string::npos)
                return false;
            return fallback;
        };
        const size_t nameAt = text.find("\"name\"");
        if (nameAt == std::string::npos)
        {
            error = "Missing name in " + path;
            return false;
        }
        const size_t firstQuote = text.find('"', nameAt + 6);
        const size_t secondQuote = firstQuote == std::string::npos ? std::string::npos : text.find('"', firstQuote + 1);
        if (firstQuote == std::string::npos || secondQuote == std::string::npos)
        {
            error = "Bad name in " + path;
            return false;
        }
        preset.name = text.substr(firstQuote + 1, secondQuote - firstQuote - 1);
        preset.sens = number("\"sens\"", 1.0);
        preset.yaw = number("\"yaw\"", 0.022);
        preset.pitch = number("\"pitch\"", 0.022);
        preset.fovScaled = boolean("\"fovScaled\"", false);
        preset.baseFOV = number("\"baseFOV\"", 0.0);
        return true;
    }
};
