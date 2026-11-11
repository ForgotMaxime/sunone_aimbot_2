#include "runtime/thread_loops.h"
#include "Game_overlay.h"

Game_overlay* gameOverlayPtr = nullptr;
std::jthread gameOverlayThread;
std::atomic<bool> gameOverlayShouldExit(false);

std::mutex g_trackerDebugMutex;
std::vector<TrackDebugInfo> g_trackerDebugTracks;
int g_trackerLockedId = -1;
