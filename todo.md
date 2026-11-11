# Sunone Aimbot 2 — TODO Améliorations

Synthèse des 5 subagents recherche (détection, capture, souris, overlay, archi).
État après merge des 5 PRs #1-#5 (fork/main `441eafa9`) + fix C++20.
Build CUDA validé (RTX 3060, CUDA 13.1, TRT 10.14) : `ai.exe` linké, **ctest 9/9 pass**.
Légende : `[x]` fait, `[~]` partiel, `[ ]` restant.

## P0 — Latence capture → infer (gain 5-20ms)

- [~] Ring SPSC triple-buffer latest-only — `LatestFrameRing` header-only ajouté, `deque<Mat>` legacy conservé, migration `capture.cpp` restante
- [x] Skip infer si `AccumulatedFrames==0` (`GpuCaptureStatus::NoPresent`)
- [x] Path GPU async : `cudaMemcpy2DFromArrayAsync` + stream dédié + ping-pong (fallback synchrone)
- [x] `ReleaseFrame()` tôt + `Map(..., DO_NOT_WAIT)` + retry
- [~] Crop+scale+BGRA→RGB+NCHW fusionné — preprocess fusionné sur même stream mais OpenCV restant, pas de kernel CUDA dédié
- [ ] Affinité adapter DDA == device CUDA, `DuplicateOutput1`, thread haute priorité + `Acquire(0)` spin
- [~] WGC `MinUpdateInterval` + pool FP16 — prep header seulement, câblage `winrt_capture.cpp` restant
- [x] Verrouillage graphs + FP16 statique batch=1 (`sanitizeInferenceNode`)
- [x] Bench e2e `--bench-e2e` QPC + CSV P50/P95/P99 (stub SendInput, à valider sur machine CUDA)

## P0 — Détection IA

- [ ] A/B YOLOv12n vs v11n vs v10n réentraîné jeu
- [x] Figer export TRT statique 320 FP16 + workspace 2Go (onnxsim manuel, cf `docs/guides/trt-export.md`)
- [~] EfficientNMS_TRT — hook détection shape-6 seulement, NMS CPU conservé
- [~] CUDA-graphs warmup `enqueueV3` avant capture — pas de profiling `trtexec`/`nsys`
- [x] Letterbox 114 + inverse exacte, conf par classe, `max_det` clamp 20-50
- [ ] Résolution ROI (320-384 scan + 640 crop) — defaults non changés
- [ ] SAHI-lite 2x2 si no-target/ROI
- [~] Tracker — hysteresis 0.35/0.15 + EMA + extrapolation fait ; ByteTrack complet (`motcpp`) non intégré
- [x] Hystérésis + EMA + extrapolation timestamp
- [ ] INT8 PTQ calibré (conditionnel, seulement si budget FPS manqué)
- [x] Depth : garder `depth_anything_v2_vits_224` async 10-15Hz (décision + doc)
- [x] Décision closed-vocab (rejeter World/RT-DETR/NAS)

## P0 — Souris / humanisation `mouse/mouse.cpp`

- [x] Deadzone + delays 30-120ms + stickiness opt-in (défaut OFF, legacy identique)
- [x] Overshoot 3-8px si dist>16px (gated)
- [x] Filtre 1€/EMA après Kalman (gated)
- [x] Fitts `MT` → maxSubsteps (gated)
- [x] Minimum-jerk smootherstep (gated)
- [x] Bruit OU/Perlin seed par node (gated, legacy défaut)
- [ ] P/PD bi-état FLICK/SETTLE + Smooth X/Y séparés
- [x] Lead adaptatif + coasting decay (gated)
- [~] Anti-recul pattern-playback — stub + doc `firmware/HUMANIZER_JITTER.md`, chemin legacy byte-identique
- [~] HW passthrough/jitter — doc host-side seulement, non validé hardware
- [ ] Anti-télémétrie (FOV dynamique, dérive session, fatigue, miss)

## P1 — Overlay / Config / UX

- [ ] Palette `Ctrl+K` fuzzy search
- [x] `ImGuiTextFilter` combos longs
- [x] Hotkeys press-to-capture + conflits + `None`
- [~] Presets JSON `presets/*.json` + migrate `[Games]` — Apply/Delete/Refresh/Save OK, pas de New/Duplicate inline
- [~] `schema_version=2` + migrate + backup atomique — pas de `config.schema.json`
- [~] VSync toggle + FPS cap + idle sleep — `FLIP_DISCARD` vs `FLIP_SEQUENTIAL` non testé
- [ ] Fonts dynamiques ImGui 1.92 + HiDPI
- [ ] Multi-langue live-switch
- [ ] Onboarding first-run + manuel cherchable
- [~] Debug Conf couleur + Infer ms + FOV ring — pas de bone numbers / sent-input visualizer
- [x] Toast save/F4 + pastille dirty

## P1 — Archi / Perf / Build

- [x] `std::jthread` + `stop_token`
- [ ] Pool work-stealing / coroutines capture→infer
- [ ] Header-units C++20 modules
- [x] `CMakePresets.json` + `vcpkg.json` + sccache (à valider au configure)
- [x] Timing-cache TRT + engine versionné + `BuilderOptimizationLevel`
- [~] NMS par classe + AVX2 CPU — pas de NMS GPU
- [ ] SPSC + pools `pmr` capture→infer→mouse
- [~] Tracy `OFF` zéro-coût + `ZoneScoped` — non compilé, tag `v0.11` à confirmer
- [ ] Modularité event-bus + plugins jeux DLL
- [~] Tests golden NMS + migrate (harnais maison) — pas de Catch2, pas de CI workflow, pas d'asan

## Ordre d'attaque — état

1. [~] Ring + skip (migration `capture.cpp` restante)
2. [x] Async CUDA + ReleaseFrame
3. [x] TRT FP16 + graphs + letterbox
4. [ ] Export 12n A/B + EfficientNMS complet
5. [~] Hystérésis fait, ByteTrack complet restant
6. [x] Délais/deadzone + overshoot + 1€ (gated OFF — activer/tester en jeu)
7. [~] Fitts + Perlin fait, P/PD restant
8. [~] Bench e2e fait, kernel préprocess GPU restant
9. [~] Presets + schema + toast faits, palette restante
10. [~] jthread + presets + Tracy + timing-cache faits, à valider par build réel
