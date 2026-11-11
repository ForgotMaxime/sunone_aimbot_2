#pragma once

// Debounced config autosave state shared by every overlay tab.
void OverlayConfig_MarkDirty();
bool OverlayConfig_IsDirty();
void OverlayConfig_TrySave(const char* filename = "config.ini");
void OverlayConfig_SaveNow(const char* filename = "config.ini");

// Toast notifications drawn inside the overlay frame.
void OverlayConfig_NotifySaved(const char* filename = "config.ini");
void OverlayConfig_NotifyReloaded();
void OverlayConfig_DrawStatusToast();
