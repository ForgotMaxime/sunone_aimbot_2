# Humanizer — Firmware Jitter Notes (host-side only, no flash required)

> Scope: `sunone_aimbot_2/firmware/` — documentation only.
> No `.ino` change, no flash procedure, no HID report format change.
> All humanizer timing/noise lives host-side (`mouse/`, `runtime/`).

## Principle

Firmware bridges (Teensy 4.1 RawHID, Arduino, RP2350, KmboxA/Net, MAKCU,
GHub, Razer, Win32) forward integer counts at whatever cadence the host
sends them. Legitimacy jitter therefore belongs in the host pipeline
(Wind synthesis → `dispatchGraphMovement` → `queueMove` → `sendMovementToDriver`),
not in device firmware. Flashing a jitter table into firmware would freeze
timing per backend and risk desync with detection delay compensation.

## Per-backend pacing (host keeps control)

| Backend | Host pacing | Notes |
|---|---|---|
| Win32 (`SendInput`) | OS coalesces; keep batches ≤ 5 substeps, ≥ 1 ms apart | No firmware involved |
| GHub / Razer | Driver applies its own smoothing; prefer small steps (≤ 10 counts) | Do not add firmware delay |
| Arduino / RP2350 / Teensy41 (serial) | Baudrate-bound; 115200 baud ≈ 1 ms/byte; batch steps per tick | Keep `move()` non-blocking |
| Teensy41 RawHID | HID poll ~1 ms; `moveMouseInSteps` already splits >127 counts | No change needed |
| KmboxNet / KmboxA / Makcu | UDP/serial bridges; network jitter dominates | Host-side OU/Perlin noise suffices |

## Recommended host jitter (matches new `humanizer_*` keys)

- Variable delays: sample uniformly in [30,120] ms per Target/Switch/Kill/
  VisCheck/Aim event (`humanizer_*_delay_ms` + `humanizer_delay_jitter_ms`).
- Correlated noise: OU (`humanizer_noise_mode=1`, theta≈2.5, sigma≈1.0) or
  Perlin1D (`mode=2`, per-node seed) instead of white uniform. Scale with
  speed via `humanizer_noise_signal_gain`.
- Wind pacing: Fitts `MT=a+b*log2(D/W+1)` drives `maxSubsteps` (1–5);
  minimum-jerk smootherstep redistributes steps without changing the sum.
- HID hygiene: never send `(0,0)`; preserve fractional carry (`windFracX/Y`,
  `recoilRemainder`); clamp batches to 32 steps / ±32767 counts (already
  enforced in `dispatchGraphMovement`).

## Anti-recoil pattern-playback (stub)

- `runtime/mouse_thread_loop.cpp` keeps the legacy `easynorecoilstrength`
  path byte-identical when `humanizer_recoil_pattern_enabled=false` (default).
- Pattern tables (future `humanizer_recoil_pattern_*`) are host-side vectors
  consumed in `resolveRecoilStrength()`; firmware only sees the resulting
  integer counts via `dispatchGraphRecoil()`.
- Do NOT implement recoil tables in `.ino`: per-weapon tables change often
  and belong in host config/graphs, not flashed firmware.

## Validation (no hardware needed)

1. `humanizer_*` all OFF → legacy trajectories (existing tests pass).
2. Enable one feature at a time; check Wind trail stays bounded
   (carry cap 120, batch ≤ 32 steps).
3. OU/Perlin: consecutive samples correlate (lag-1 autocorr > 0.5), uniform
   does not.
4. Overshoot: moves >16 px overshoot 3–8 px then return once residual <
   (1−0.85)·initial.
5. Delays: first publication delayed by sampled [30,120] ms; kill path coasts
   instead of snapping.
