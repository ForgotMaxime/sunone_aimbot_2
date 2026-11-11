#pragma once

// Zero-cost Tracy profiling hooks (AIMBOT_TRACY=OFF by default).
// Enable with -DAIMBOT_TRACY=ON: CMake fetches wolfpld/tracy and defines
// AIMBOT_HAS_TRACY + TRACY_ENABLE + TRACY_ON_DEMAND, linking TracyClient.
// With the option OFF these macros compile to nothing: no new dependency,
// no runtime cost, no functional change.
#ifdef AIMBOT_HAS_TRACY
#include "tracy/Tracy.hpp"
#define AIMBOT_ZONE ZoneScoped
#define AIMBOT_FRAME_MARK FrameMark
#else
#define AIMBOT_ZONE do {} while (0)
#define AIMBOT_FRAME_MARK do {} while (0)
#endif
