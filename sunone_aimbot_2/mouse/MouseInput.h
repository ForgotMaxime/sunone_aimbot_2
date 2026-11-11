#ifndef MOUSE_INPUT_H
#define MOUSE_INPUT_H

#include <memory>
#include <optional>
#include <string>

class Arduino;
class Config;
class GhubMouse;
class KmboxAConnection;
class KmboxNetConnection;
class MakcuConnection;
class RP2350;
class RzctlMouse;
class Teensy41RawHid;

enum class MouseInputMethod
{
    Win32,
    GHub,
    Razer,
    Arduino,
    RP2350,
    Teensy41,
    Teensy41Hid,
    KmboxNet,
    KmboxA,
    Makcu
};

std::optional<MouseInputMethod> ParseMouseInputMethod(const std::string& method);
std::string MouseInputMethodName(MouseInputMethod method);

// Humanizer (Sunone Aimbot 2) — conservative defaults shared by mouse.cpp
// and runtime/mouse_thread_loop.cpp. All features default to OFF (0/false)
// so existing graphs keep byte-identical behavior unless a
// "humanizer_*" node parameter or Config key opts in.
struct HumanizerDefaults
{
    // Deadzone pivot (px in capture space). 0 = disabled.
    static constexpr double deadzonePx = 0.0;
    // Target/Switch/Kill/VisCheck/Aim delays. 0 = disabled (immediate).
    // When enabled, sample uniformly in [30,120]ms +/- jitter.
    static constexpr double delayMinMs = 30.0;
    static constexpr double delayMaxMs = 120.0;
    // Target stickiness (anti-pullback hysteresis).
    static constexpr double stickinessMarginPx = 25.0;
    static constexpr int stickinessHoldFrames = 3;
    static constexpr double switchCooldownMs = 120.0;
    // Overshoot: applied only when dist > distPx, amplitude in [minPx,maxPx],
    // return scheduled once progress t > returnT.
    static constexpr double overshootMinPx = 3.0;
    static constexpr double overshootMaxPx = 8.0;
    static constexpr double overshootDistPx = 16.0;
    static constexpr double overshootReturnT = 0.85;
    // One-Euro / EMA post-filter after AimKalman2D.
    static constexpr double oneEuroFcMin = 1.0;
    static constexpr double oneEuroBeta = 0.05;
    static constexpr double oneEuroDCutoff = 1.0;
    // Fitts law MT = a + b*log2(D/W+1).
    static constexpr double fittsA = 0.01;
    static constexpr double fittsB = 0.03;
    static constexpr double fittsW = 24.0;
    // Correlated noise: 0=legacy uniform, 1=Ornstein-Uhlenbeck, 2=Perlin1D.
    static constexpr int noiseModeLegacy = 0;
    static constexpr double ouTheta = 2.5;
    static constexpr double ouSigma = 1.0;
    // Adaptive lead + coasting.
    static constexpr double leadGain = 0.0004;
    static constexpr double leadMaxScale = 1.6;
    static constexpr int coastFrames = 6;
    static constexpr double coastDecay = 0.90;
};

class IMouseInput
{
public:
    virtual ~IMouseInput() = default;

    virtual const char* name() const = 0;
    virtual bool isOpen() const = 0;
    virtual bool move(int dx, int dy) = 0;
    virtual bool leftDown() = 0;
    virtual bool leftUp() = 0;
    virtual bool hasPhysicalButtonState() const { return false; }
    virtual bool keyPressed(const std::string& keyName) { (void)keyName; return false; }
    virtual bool aimingActive() const { return false; }
    virtual bool shootingActive() const { return false; }
    virtual bool zoomingActive() const { return false; }

    virtual Arduino* arduino() { return nullptr; }
    virtual RP2350* rp2350() { return nullptr; }
    virtual GhubMouse* ghub() { return nullptr; }
    virtual RzctlMouse* razer() { return nullptr; }
    virtual KmboxNetConnection* kmboxNet() { return nullptr; }
    virtual KmboxAConnection* kmboxA() { return nullptr; }
    virtual MakcuConnection* makcu() { return nullptr; }
    virtual Teensy41RawHid* teensy41RawHid() { return nullptr; }
};

std::unique_ptr<IMouseInput> CreateMouseInputDevice(const Config& config);

#endif // MOUSE_INPUT_H
