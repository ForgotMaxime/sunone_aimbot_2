#pragma once

// C++20 portability: std::filesystem::path::u8string() returns a char8_t
// based std::u8string since C++20 (it was std::string in C++17), so it no
// longer converts implicitly to std::string. These helpers byte-copy the
// UTF-8 sequence back into std::string, preserving the exact historical
// (C++17) narrow-string behavior for model paths and UI labels.

#include <filesystem>
#include <string>

namespace aimbot
{
inline std::string fsUtf8(const std::filesystem::path& path)
{
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

inline std::string fsUtf8Generic(const std::filesystem::path& path)
{
    const auto utf8 = path.generic_u8string();
    return std::string(utf8.begin(), utf8.end());
}
} // namespace aimbot
