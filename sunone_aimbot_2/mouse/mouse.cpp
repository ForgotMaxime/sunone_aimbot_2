#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <random>
#include <limits>

#include "mouse.h"
#include "capture.h"
#include "detection_buffer.h"
#include "runtime/input_state.h"

namespace
{
double currentFrameIntervalSec()
{
    double fps = static_cast<double>(captureFps.load());
    if (fps <= 0.0) fps = 60.0;

    fps = std::clamp(fps, 15.0, 500.0);
    return 1.0 / fps;
}

template <typename States>
bool hasGraphStateCapacity(const States& states, int nodeKey)
{
    constexpr int graphStride = 1000001;
    if (nodeKey <= 0 || nodeKey >= graphStride * static_cast<int>(NodeGraph::MaxGraphs))
        return false;
    const int graph = nodeKey / graphStride;
    std::size_t count = 0;
    for (const auto& entry : states)
        if (entry.first / graphStride == graph)
            ++count;
    return count < NodeGraph::MaxNodes;
}
} // namespace

// Humanizer helpers (conservative: every feature defaults to OFF and is only
// enabled by an explicit "humanizer_*" node parameter > 0).
namespace humanizer
{
struct WindOpts
{
    bool overshootEnabled = false;
    double overshootMinPx = 3.0, overshootMaxPx = 8.0;
    double overshootDistPx = 16.0, overshootReturnT = 0.85;
    bool fittsEnabled = false;
    double fittsA = 0.05, fittsB = 0.12, fittsW = 24.0;
    bool jerkEnabled = false;
    int noiseMode = 0; // 0 legacy uniform, 1 OU, 2 Perlin1D
    double ouTheta = 2.5, ouSigma = 1.0, noiseSignalGain = 0.0;
    double deadzonePx = 0.0;
};

inline double nodeParam(const NodeGraph::Node& node, const char* key, double fallback)
{
    const double v = NodeGraph::parameter(node, key);
    // parameter() returns 0 for unknown keys: treat exact 0 as "unset" so
    // legacy graphs keep legacy behavior; explicit 0 stays disabled anyway.
    if (v == 0.0) return fallback;
    return v;
}

inline bool nodeFlag(const NodeGraph::Node& node, const char* key)
{
    return NodeGraph::parameter(node, key) != 0.0;
}

inline WindOpts windOptsFromNode(const NodeGraph::Node& node)
{
    WindOpts o;
    o.overshootEnabled = nodeFlag(node, "humanizer_overshoot");
    o.overshootMinPx = std::clamp(nodeParam(node, "humanizer_overshoot_min_px", 3.0), 0.0, 32.0);
    o.overshootMaxPx = std::clamp(nodeParam(node, "humanizer_overshoot_max_px", 8.0), 0.0, 32.0);
    if (o.overshootMaxPx < o.overshootMinPx) std::swap(o.overshootMinPx, o.overshootMaxPx);
    o.overshootDistPx = std::clamp(nodeParam(node, "humanizer_overshoot_dist_px", 16.0), 0.0, 512.0);
    o.overshootReturnT = std::clamp(nodeParam(node, "humanizer_overshoot_return_t", 0.85), 0.0, 1.0);
    o.fittsEnabled = nodeFlag(node, "humanizer_fitts");
    o.fittsA = std::clamp(nodeParam(node, "humanizer_fitts_a", 0.01), 0.0, 2.0);
    o.fittsB = std::clamp(nodeParam(node, "humanizer_fitts_b", 0.03), 0.0, 2.0);
    o.fittsW = std::clamp(nodeParam(node, "humanizer_fitts_W", 24.0), 1.0, 1024.0);
    o.jerkEnabled = nodeFlag(node, "humanizer_jerk");
    o.noiseMode = static_cast<int>(std::clamp(NodeGraph::parameter(node, "humanizer_noise_mode"), 0.0, 2.0));
    o.ouTheta = std::clamp(nodeParam(node, "humanizer_noise_theta", 2.5), 0.01, 20.0);
    o.ouSigma = std::clamp(nodeParam(node, "humanizer_noise_sigma", 1.0), 0.0, 8.0);
    o.noiseSignalGain = std::clamp(NodeGraph::parameter(node, "humanizer_noise_signal_gain"), 0.0, 4.0);
    o.deadzonePx = std::clamp(NodeGraph::parameter(node, "humanizer_deadzone_px"), 0.0, 256.0);
    return o;
}

// Fitts law: MT = a + b * log2(D/W + 1). Returns movement time in seconds.
inline double fittsMT(double distPx, double wPx, double a, double b)
{
    if (!(distPx > 0.0) || !(wPx > 0.0)) return 0.0;
    return std::max(0.0, a + b * std::log2(distPx / wPx + 1.0));
}

// Minimum-jerk smootherstep velocity weight at normalized time t in [0,1]:
// s(t) = 10t^3 - 15t^4 + 6t^5 ; w(t) = ds/dt normalized.
inline double jerkVelocity(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    const double t2 = t * t;
    // ds/dt = 30t^2 - 60t^3 + 30t^4, mean over [0,1] is 1 -> normalized already.
    return 30.0 * t2 - 60.0 * t2 * t + 30.0 * t2 * t2;
}

// Ornstein-Uhlenbeck step: x += theta*(0-x)*dt + sigma*sqrt(dt)*N(0,1).
inline double ouStep(double x, double theta, double sigma, double dt,
    std::mt19937& rng, std::normal_distribution<double>& gauss)
{
    x += -theta * x * dt + sigma * std::sqrt(std::max(1e-6, dt)) * gauss(rng);
    return std::clamp(x, -4.0, 4.0);
}

// Deterministic Perlin-like 1D gradient noise in [-1,1], seeded per node.
inline double perlin1D(double t, std::uint32_t seed)
{
    const long long xi = static_cast<long long>(std::floor(t));
    const double xf = t - static_cast<double>(xi);
    auto hash = [&](long long n) -> double {
        std::uint64_t h = static_cast<std::uint64_t>(n * 374761393LL + seed * 668265263LL);
        h = (h ^ (h >> 13)) * 1274126177ULL;
        h ^= h >> 16;
        return static_cast<double>(h % 1000000ULL) / 500000.0 - 1.0;
    };
    const double g0 = hash(xi), g1 = hash(xi + 1);
    const double u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0);
    return g0 + (g1 - g0) * u;
}

// One-Euro filter state per prediction node.
struct OneEuroState
{
    bool init = false;
    double x = 0.0, y = 0.0;
    double dx = 0.0, dy = 0.0;
    std::chrono::steady_clock::time_point t{};
};

inline double oneEuroAlpha(double fc, double dt)
{
    const double tau = 1.0 / (2.0 * 6.28318530717958647692 * std::max(1e-3, fc));
    return 1.0 / (1.0 + tau / std::max(1e-4, dt));
}

inline std::pair<double, double> oneEuroFilter(OneEuroState& s, double mx, double my,
    double dt, double fcMin, double beta, double dCutoff)
{
    const auto now = std::chrono::steady_clock::now();
    if (!s.init)
    {
        s.init = true; s.x = mx; s.y = my; s.dx = s.dy = 0.0; s.t = now;
        return { mx, my };
    }
    const double rawDx = (mx - s.x) / std::max(1e-4, dt);
    const double rawDy = (my - s.y) / std::max(1e-4, dt);
    const double aD = oneEuroAlpha(dCutoff, dt);
    s.dx += (rawDx - s.dx) * aD;
    s.dy += (rawDy - s.dy) * aD;
    const double speed = std::hypot(s.dx, s.dy);
    const double fc = fcMin + beta * speed;
    const double a = oneEuroAlpha(fc, dt);
    s.x += (mx - s.x) * a;
    s.y += (my - s.y) * a;
    s.t = now;
    return { s.x, s.y };
}

// Coasting state per prediction node: decay velocity instead of hard reset.
struct CoastState
{
    int coastLeft = 0;
    double px = 0.0, py = 0.0, vx = 0.0, vy = 0.0;
};

} // namespace humanizer

namespace
{
// File-static humanizer state keeps mouse.h untouched (strict scope).
std::unordered_map<int, humanizer::OneEuroState> g_oneEuroStates;
std::unordered_map<int, humanizer::CoastState> g_coastStates;
std::unordered_map<int, std::pair<double, double>> g_ouStates;
std::unordered_map<int, std::pair<double, double>> g_overshootDebt;
std::unordered_map<int, double> g_overshootInitial;
std::unordered_map<int, double> g_perlinClock;
thread_local humanizer::WindOpts t_windOpts;
thread_local int t_windNodeKey = 0;
thread_local bool t_hasWindOpts = false;
struct WindOptsGuard
{
    WindOptsGuard(const humanizer::WindOpts& o, int key) { t_windOpts = o; t_windNodeKey = key; t_hasWindOpts = true; }
    ~WindOptsGuard() { t_hasWindOpts = false; t_windNodeKey = 0; }
};
} // namespace

MouseThread::MouseThread(int resolution,IMouseInput* mouseInputDevice)
    :screen_width(std::max(1,resolution)),screen_height(std::max(1,resolution)),
     center_x(screen_width*0.5),center_y(screen_height*0.5),mouseInput(mouseInputDevice)
{
    wind_mouse_enabled=true;wind_G=9;wind_W=3;wind_M=10;wind_D=12;
    resetWindState();clearWindDebugTrail();
    detectionGeneration=detectionBuffer.currentGeneration();
    moveWorker=std::thread(&MouseThread::moveWorkerLoop,this);
}
void MouseThread::setCaptureResolution(int resolution)
{
    resetPrediction();
    screen_width=screen_height=std::max(1,resolution);center_x=screen_width*0.5;center_y=screen_height*0.5;
    clearWindDebugTrail();
}

MouseThread::~MouseThread()
{
    stop();
}

void MouseThread::requestStop()
{
    {
        std::lock_guard<std::mutex> lock(queueMtx);
        workerStop = true;
        ++queueGeneration;
        while (!moveQueue.empty())
            moveQueue.pop();
    }
    queueCv.notify_all();
}

void MouseThread::stop()
{
    requestStop();
    if (moveWorker.joinable())
        moveWorker.join();
    setMouseInput(nullptr);
}

void MouseThread::queueMove(int dx, int dy, double pixelScaleX, double pixelScaleY)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    std::lock_guard lg(queueMtx);
    if (workerStop)
        return;
    if (moveQueue.size() >= queueLimit) moveQueue.pop();
    moveQueue.push({ dx, dy, detectionGeneration, queueGeneration, pixelScaleX, pixelScaleY });
    queueCv.notify_one();
}

void MouseThread::moveWorkerLoop()
{
    try
    {
        for (;;)
        {
            std::unique_lock ul(queueMtx);
            queueCv.wait(ul, [&] { return workerStop || !moveQueue.empty(); });
            if (workerStop)
                break;

            Move m = moveQueue.front();
            moveQueue.pop();
            ul.unlock();

            std::lock_guard<std::mutex> dispatchLock(dispatchMutex);
            {
                std::lock_guard<std::mutex> queueLock(queueMtx);
                if (workerStop || m.queueGeneration != queueGeneration)
                    continue;
            }
            sendMovementToDriver(m.dx, m.dy, m.detectionGeneration, m.pixelScaleX, m.pixelScaleY);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] Move worker crashed: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] Move worker crashed: unknown exception." << std::endl;
    }
}

NodeGraph::Movement MouseThread::buildWindSteps(int dx, int dy)
{
    NodeGraph::Movement movement;
    if (dx == 0 && dy == 0)
        return movement;
    movement.steps.reserve(5);

    windCarryX += static_cast<double>(dx);
    windCarryY += static_cast<double>(dy);

    const double baseG = std::clamp(wind_G, 0.05, 50.0);
    const double baseW = std::clamp(wind_W, 0.0, 80.0);
    const double baseM = std::max(1.0, wind_M);
    const double baseD = std::max(1.0, wind_D);
    if (!std::isfinite(baseG) || !std::isfinite(baseW) ||
        !std::isfinite(baseM) || !std::isfinite(baseD))
    {
        resetWindState();
        return movement;
    }

    // Humanizer options for this synthesis (legacy defaults when no guard).
    const humanizer::WindOpts opts = t_hasWindOpts ? t_windOpts : humanizer::WindOpts{};
    const int nodeKey = t_windNodeKey;

    // Overshoot: inject 3-8px along the aim direction when dist > 16px, then
    // schedule the opposite correction once progress t > returnT. The return
    // is applied when the residual drops below (1-returnT)*initial.
    if (opts.overshootEnabled && nodeKey != 0)
    {
        const double preMag = std::hypot(windCarryX, windCarryY);
        auto debtIt = g_overshootDebt.find(nodeKey);
        auto initIt = g_overshootInitial.find(nodeKey);
        if (debtIt != g_overshootDebt.end() && initIt != g_overshootInitial.end())
        {
            const double threshold = initIt->second * (1.0 - opts.overshootReturnT);
            if (preMag <= threshold || preMag < 1.0)
            {
                windCarryX += debtIt->second.first;
                windCarryY += debtIt->second.second;
                g_overshootDebt.erase(debtIt);
                g_overshootInitial.erase(initIt);
            }
        }
        else if (preMag > opts.overshootDistPx && preMag > 1e-8)
        {
            std::uniform_real_distribution<double> ampDist(opts.overshootMinPx, opts.overshootMaxPx);
            const double amp = std::clamp(ampDist(windRng), opts.overshootMinPx, opts.overshootMaxPx);
            const double ox = windCarryX / preMag * amp;
            const double oy = windCarryY / preMag * amp;
            windCarryX += ox;
            windCarryY += oy;
            g_overshootDebt[nodeKey] = { -ox, -oy };
            g_overshootInitial[nodeKey] = preMag + amp;
        }
    }

    std::uniform_real_distribution<double> noiseDist(-1.0, 1.0);
    std::normal_distribution<double> gaussDist(0.0, 1.0);
    std::uniform_real_distribution<double> clipDist(0.55, 1.0);
    constexpr double twoPi = 6.28318530717958647692;

    int maxSubsteps = std::clamp(static_cast<int>(std::hypot(windCarryX, windCarryY) * 0.24) + 1, 1, 5);
    if (opts.fittsEnabled)
    {
        // Fitts law drives the Wind duration: MT=a+b*log2(D/W+1).
        const double distPx = std::hypot(windCarryX, windCarryY);
        const double mt = humanizer::fittsMT(distPx, opts.fittsW, opts.fittsA, opts.fittsB);
        if (mt > 0.0)
            maxSubsteps = std::clamp(static_cast<int>(std::ceil(mt / currentFrameIntervalSec())), 1, 5);
    }

    for (int i = 0; i < maxSubsteps; ++i)
    {
        const double dist = std::hypot(windCarryX, windCarryY);
        const double velMag = std::hypot(windVelX, windVelY);

        if (dist < 0.20 && velMag < 0.12)
            break;

        const double normDist = std::clamp(dist / baseD, 0.0, 1.0);
        const double pullGain = baseG * (0.25 + 0.75 * normDist);
        const double noiseAmp = baseW * (0.15 + 0.85 * normDist);

        double pullX = 0.0;
        double pullY = 0.0;
        if (dist > 1e-8)
        {
            pullX = windCarryX / dist * pullGain;
            pullY = windCarryY / dist * pullGain;
        }

        windPatternRateA = std::clamp(windPatternRateA + noiseDist(windRng) * 0.004, 0.025, 0.280);
        windPatternRateB = std::clamp(windPatternRateB + noiseDist(windRng) * 0.004, 0.025, 0.280);

        const double stepTempo = 0.20 + 0.95 * normDist;
        windPatternPhaseA += windPatternRateA * stepTempo;
        windPatternPhaseB += windPatternRateB * stepTempo;
        if (windPatternPhaseA > twoPi) windPatternPhaseA = std::fmod(windPatternPhaseA, twoPi);
        if (windPatternPhaseB > twoPi) windPatternPhaseB = std::fmod(windPatternPhaseB, twoPi);

        const double oscAX = std::sin(windPatternPhaseA);
        const double oscBX = std::sin(windPatternPhaseB + 1.61803398875);
        const double oscAY = std::cos(windPatternPhaseA * 0.79 + 0.35);
        const double oscBY = std::cos(windPatternPhaseB * 1.17 - 0.48);

        const double patternAmp = baseW * (0.05 + 0.55 * normDist);
        const double patternTargetX = (oscAX + 0.58 * oscBX) * patternAmp;
        const double patternTargetY = (oscAY + 0.58 * oscBY) * patternAmp;
        const double patternBlend = 0.12 + 0.20 * normDist;
        windPatternX = windPatternX * (1.0 - patternBlend) + patternTargetX * patternBlend;
        windPatternY = windPatternY * (1.0 - patternBlend) + patternTargetY * patternBlend;

        // Correlated + signal-dependent noise. Mode 0 keeps the legacy
        // uniform path byte-identical; modes 1/2 add OU/Perlin1D correlation
        // (seeded per nodeKey) scaled by speed when noiseSignalGain > 0.
        double sampleX = noiseDist(windRng);
        double sampleY = noiseDist(windRng);
        if (opts.noiseMode == 1 && nodeKey != 0)
        {
            auto& ou = g_ouStates[nodeKey]; // value-initialized to (0,0)
            const double dt = currentFrameIntervalSec();
            ou.first = humanizer::ouStep(ou.first, opts.ouTheta, opts.ouSigma, dt, windRng, gaussDist);
            ou.second = humanizer::ouStep(ou.second, opts.ouTheta, opts.ouSigma, dt, windRng, gaussDist);
            sampleX = ou.first;
            sampleY = ou.second;
        }
        else if (opts.noiseMode == 2 && nodeKey != 0)
        {
            double& clock = g_perlinClock[nodeKey];
            clock += (0.35 + 0.65 * normDist);
            sampleX = humanizer::perlin1D(clock, static_cast<std::uint32_t>(nodeKey * 2654435761u + 97u));
            sampleY = humanizer::perlin1D(clock * 1.31 + 17.7, static_cast<std::uint32_t>(nodeKey * 40503u + 13u));
        }
        const double signalScale = 1.0 + opts.noiseSignalGain * std::clamp(velMag / std::max(1.0, baseM), 0.0, 2.0);
        windNoiseX = windNoiseX * 0.72 + sampleX * noiseAmp * signalScale * 0.28;
        windNoiseY = windNoiseY * 0.72 + sampleY * noiseAmp * signalScale * 0.28;

        const double windForceX = windNoiseX + windPatternX * 0.42;
        const double windForceY = windNoiseY + windPatternY * 0.42;

        const double drag = 0.82 + (1.0 - normDist) * 0.10;
        windVelX = windVelX * drag + pullX + windForceX;
        windVelY = windVelY * drag + pullY + windForceY;

        const double vCap = std::max(0.65, baseM * (0.30 + 0.70 * normDist));
        const double newVelMag = std::hypot(windVelX, windVelY);
        if (newVelMag > vCap)
        {
            const double clip = vCap * clipDist(windRng);
            windVelX = (windVelX / newVelMag) * clip;
            windVelY = (windVelY / newVelMag) * clip;
        }

        windFracX += windVelX;
        windFracY += windVelY;

        int stepX = static_cast<int>(std::clamp(std::round(windFracX), -32767.0, 32767.0));
        int stepY = static_cast<int>(std::clamp(std::round(windFracY), -32767.0, 32767.0));
        if (stepX == 0 && stepY == 0)
            continue;

        windFracX -= static_cast<double>(stepX);
        windFracY -= static_cast<double>(stepY);
        windCarryX -= static_cast<double>(stepX);
        windCarryY -= static_cast<double>(stepY);
        movement.steps.push_back({stepX, stepY});
    }

    const double carryCap = 120.0;
    const double finalCarryMag = std::hypot(windCarryX, windCarryY);
    if (finalCarryMag > carryCap)
    {
        const double s = carryCap / finalCarryMag;
        windCarryX *= s;
        windCarryY *= s;
    }
    if (opts.jerkEnabled && movement.steps.size() >= 2)
    {
        // Minimum-jerk smootherstep redistribution: timing follows the
        // bell-shaped jerk velocity profile while the summed displacement
        // stays identical (rounding drift fixed on the last step).
        const std::size_t n = movement.steps.size();
        long long totalX = 0, totalY = 0;
        for (const auto& s : movement.steps) { totalX += s.dx; totalY += s.dy; }
        std::vector<double> w(n);
        double wSum = 0.0;
        for (std::size_t k = 0; k < n; ++k)
        {
            w[k] = std::max(1e-6, humanizer::jerkVelocity((static_cast<double>(k) + 0.5) / static_cast<double>(n)));
            wSum += w[k];
        }
        long long accX = 0, accY = 0;
        for (std::size_t k = 0; k < n; ++k)
        {
            if (k + 1 == n)
            {
                movement.steps[k].dx = static_cast<int>(totalX - accX);
                movement.steps[k].dy = static_cast<int>(totalY - accY);
            }
            else
            {
                const int nx = static_cast<int>(std::lround(static_cast<double>(totalX) * w[k] / wSum));
                const int ny = static_cast<int>(std::lround(static_cast<double>(totalY) * w[k] / wSum));
                movement.steps[k].dx = nx;
                movement.steps[k].dy = ny;
                accX += nx;
                accY += ny;
            }
        }
    }
    return movement;
}

void MouseThread::resetWindState()
{
    graphWindStates.clear();
    resetWindScalars();
}

void MouseThread::resetWindScalars()
{
    constexpr double twoPi = 6.28318530717958647692;
    std::uniform_real_distribution<double> phaseDist(0.0, twoPi);
    std::uniform_real_distribution<double> rateDist(0.04, 0.16);

    windCarryX = 0.0;
    windCarryY = 0.0;
    windVelX = 0.0;
    windVelY = 0.0;
    windNoiseX = 0.0;
    windNoiseY = 0.0;
    windFracX = 0.0;
    windFracY = 0.0;
    windPatternX = 0.0;
    windPatternY = 0.0;
    windPatternPhaseA = phaseDist(windRng);
    windPatternPhaseB = phaseDist(windRng);
    windPatternRateA = rateDist(windRng);
    windPatternRateB = rateDist(windRng);
}

MouseThread::WindNodeState MouseThread::captureWindState() const
{
    WindNodeState state;
    state.carryX = windCarryX; state.carryY = windCarryY;
    state.velocityX = windVelX; state.velocityY = windVelY;
    state.noiseX = windNoiseX; state.noiseY = windNoiseY;
    state.fractionX = windFracX; state.fractionY = windFracY;
    state.patternX = windPatternX; state.patternY = windPatternY;
    state.phaseA = windPatternPhaseA; state.phaseB = windPatternPhaseB;
    state.rateA = windPatternRateA; state.rateB = windPatternRateB;
    state.rng = windRng;
    return state;
}

void MouseThread::restoreWindState(const WindNodeState& state)
{
    windCarryX = state.carryX; windCarryY = state.carryY;
    windVelX = state.velocityX; windVelY = state.velocityY;
    windNoiseX = state.noiseX; windNoiseY = state.noiseY;
    windFracX = state.fractionX; windFracY = state.fractionY;
    windPatternX = state.patternX; windPatternY = state.patternY;
    windPatternPhaseA = state.phaseA; windPatternPhaseB = state.phaseB;
    windPatternRateA = state.rateA; windPatternRateB = state.rateB;
    windRng = state.rng;
}

void MouseThread::appendWindDebugStep(int dx, int dy, double pixelScaleX, double pixelScaleY)
{
    if (dx == 0 && dy == 0)
        return;

    const auto delta=std::make_pair(dx*pixelScaleX,dy*pixelScaleY);
    if(!std::isfinite(delta.first)||!std::isfinite(delta.second))return;
    double deltaPxX = delta.first;
    double deltaPxY = delta.second;
    if (std::abs(deltaPxX) < 1e-8 && std::abs(deltaPxY) < 1e-8)
        return;
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    if (windDebugTrail.empty())
    {
        windDebugCursorX = center_x;
        windDebugCursorY = center_y;
        windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });
    }

    windDebugCursorX += deltaPxX;
    windDebugCursorY += deltaPxY;
    windDebugTrail.push_back({ windDebugCursorX, windDebugCursorY, now });

    constexpr size_t maxTrailPoints = 220;
    while (windDebugTrail.size() > maxTrailPoints)
        windDebugTrail.pop_front();
}

void MouseThread::pruneWindDebugTrailLocked(const std::chrono::steady_clock::time_point& now)
{
    constexpr auto windTrailLifetime = std::chrono::milliseconds(900);
    while (!windDebugTrail.empty() && (now - windDebugTrail.front().t) > windTrailLifetime)
        windDebugTrail.pop_front();
}

void MouseThread::recordMotionCompensationStep(int dx, int dy, double pixelScaleX, double pixelScaleY)
{
    if (dx == 0 && dy == 0)
        return;

    const auto delta=std::make_pair(dx*pixelScaleX,dy*pixelScaleY);
    if(!std::isfinite(delta.first)||!std::isfinite(delta.second))return;
    if (std::abs(delta.first) < 1e-8 && std::abs(delta.second) < 1e-8)
        return;

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(motionCompensationMutex);
    pruneMotionCompensationTrailLocked(now);
    motionCompensationTrail.push_back({ delta.first, delta.second, now });

    constexpr size_t maxSamples = 512;
    while (motionCompensationTrail.size() > maxSamples)
        motionCompensationTrail.pop_front();
}

void MouseThread::pruneMotionCompensationTrailLocked(const std::chrono::steady_clock::time_point& now)
{
    constexpr auto motionTrailLifetime = std::chrono::seconds(2);
    while (!motionCompensationTrail.empty() && (now - motionCompensationTrail.front().t) > motionTrailLifetime)
        motionCompensationTrail.pop_front();
}

std::pair<double, double> MouseThread::getMotionCompensationSince(
    std::chrono::steady_clock::time_point since) const
{
    if (since.time_since_epoch().count() == 0)
        return { 0.0, 0.0 };

    double x = 0.0;
    double y = 0.0;
    std::lock_guard<std::mutex> lock(motionCompensationMutex);
    for (const auto& sample : motionCompensationTrail)
    {
        if (sample.t >= since)
        {
            x += sample.x;
            y += sample.y;
        }
    }

    return { x, y };
}

std::pair<double, double> MouseThread::graphPredict(int nodeKey, double pointX, double pointY,
    std::chrono::steady_clock::time_point observationTime, const NodeGraph::Node& node,
    double lookahead, bool reset)
{
    const auto invalid = std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(pointX) || !std::isfinite(pointY) || !std::isfinite(lookahead)) return invalid;
    const auto motion=graphEstimateMotion(nodeKey,{pointX,pointY},observationTime,node,reset);
    if(observationTime.time_since_epoch().count()==0)observationTime=std::chrono::steady_clock::now();
    const double delay=std::clamp(std::chrono::duration<double>(std::chrono::steady_clock::now()-observationTime).count(),0.0,0.35);
    if(NodeGraph::parameter(node,"kalman_compensate_detection_delay")!=0)lookahead+=delay;
    lookahead=std::clamp(lookahead+NodeGraph::parameter(node,"kalman_additional_prediction_ms")*0.001,0.0,1.5);
    if (NodeGraph::parameter(node, "humanizer_lead_adaptive") != 0.0)
    {
        // Adaptive lead: scale lookahead with target speed so fast strafes get
        // more lead and slow/steady targets keep the base timing.
        const double speed = std::hypot(motion.velocity.x, motion.velocity.y);
        const double gain = humanizer::nodeParam(node, "humanizer_lead_gain", 0.0004);
        const double maxScale = std::clamp(humanizer::nodeParam(node, "humanizer_lead_max_scale", 1.6), 1.0, 3.0);
        lookahead = std::clamp(lookahead * (1.0 + speed * std::max(0.0, gain)), 0.0, 1.5 * maxScale);
        lookahead = std::clamp(lookahead, 0.0, 1.5);
    }
    const auto state=graphPredictionStates.find(nodeKey);if(state!=graphPredictionStates.end())state->second.lookaheadSec=lookahead;
    const auto result=NodeGraph::projectPoint(motion.position,motion.velocity,lookahead,motion.damping,motion.ready);
    return {result.x,result.y};
}
NodeGraph::MotionData MouseThread::graphEstimateMotion(int nodeKey,NodeGraph::Point point,
    std::chrono::steady_clock::time_point observationTime,const NodeGraph::Node& node,bool reset)
{
    NodeGraph::MotionData invalid;invalid.damping=std::numeric_limits<double>::quiet_NaN();
    if(!std::isfinite(point.x)||!std::isfinite(point.y))return invalid;
    auto found = graphPredictionStates.find(nodeKey);
    if (found == graphPredictionStates.end())
    {
        if (!hasGraphStateCapacity(graphPredictionStates,nodeKey)) return invalid;
        found = graphPredictionStates.emplace(nodeKey,PredictionNodeState{}).first;
    }
    auto& state = found->second;
    const auto now = std::chrono::steady_clock::now();
    if (observationTime.time_since_epoch().count()==0) observationTime=now;
    const double gap = state.time.time_since_epoch().count()==0 ? 0.0 :
        std::chrono::duration<double>(observationTime-state.time).count();
    const double timeout = NodeGraph::parameter(node,"kalman_reset_timeout_sec");
    aim::AimKalmanSettings settings;
    settings.enabled = NodeGraph::parameter(node,"kalman_enabled")!=0;
    settings.process_noise_position = NodeGraph::parameter(node,"kalman_process_noise_position");
    settings.process_noise_velocity = NodeGraph::parameter(node,"kalman_process_noise_velocity");
    settings.measurement_noise = NodeGraph::parameter(node,"kalman_measurement_noise");
    settings.velocity_damping = NodeGraph::parameter(node,"kalman_velocity_damping");
    settings.max_velocity = NodeGraph::parameter(node,"kalman_max_velocity");
    settings.warmup_frames = static_cast<int>(NodeGraph::parameter(node,"kalman_warmup_frames"));
    state.kalman.setSettings(settings);
    // Humanizer coasting: instead of a hard reset on timeout, coast for N
    // frames with exponential velocity decay. Disabled by default.
    const bool coastEnabled = NodeGraph::parameter(node, "humanizer_coast") != 0.0;
    const int coastFrames = std::clamp(static_cast<int>(NodeGraph::parameter(node, "humanizer_coast_frames") == 0.0
        ? 6.0 : NodeGraph::parameter(node, "humanizer_coast_frames")), 0, 60);
    const double coastDecay = std::clamp(humanizer::nodeParam(node, "humanizer_coast_decay", 0.90), 0.5, 0.999);
    if (reset || gap<0 || (gap>timeout && !coastEnabled)) state.kalman.reset();
    if (gap > timeout && coastEnabled && !reset && gap >= 0)
    {
        auto& coast = g_coastStates[nodeKey];
        if (coast.coastLeft <= 0 && state.kalman.initialized())
        {
            const auto pos = state.kalman.position();
            const auto vel = state.kalman.velocity();
            coast = { coastFrames, pos.first, pos.second, vel.first, vel.second };
        }
        if (coast.coastLeft > 0)
        {
            const double dt = gap>0 ? std::clamp(gap,1.0/500.0,0.25) : currentFrameIntervalSec();
            coast.px += coast.vx * dt;
            coast.py += coast.vy * dt;
            coast.vx *= coastDecay;
            coast.vy *= coastDecay;
            --coast.coastLeft;
            state.time = observationTime;
            state.lookaheadSec = 0;
            // Feed the coasted position through the 1E filter below so the
            // output stays continuous when observations resume.
            double fx = coast.px, fy = coast.py;
            if (NodeGraph::parameter(node, "humanizer_oneeuro") != 0.0)
            {
                auto& f = g_oneEuroStates[nodeKey];
                const double fcMin = humanizer::nodeParam(node, "humanizer_oneeuro_fcmin", 1.0);
                const double beta = std::clamp(NodeGraph::parameter(node, "humanizer_oneeuro_beta"), 0.0, 5.0);
                const double dcut = humanizer::nodeParam(node, "humanizer_oneeuro_dcutoff", 1.0);
                const auto filtered = humanizer::oneEuroFilter(f, fx, fy, dt,
                    std::clamp(fcMin, 0.05, 20.0), beta, std::clamp(dcut, 0.05, 20.0));
                fx = filtered.first; fy = filtered.second;
            }
            return { { fx, fy }, { coast.vx, coast.vy },
                state.kalman.settings().velocity_damping, true, true };
        }
        state.kalman.reset();
    }
    else if (coastEnabled)
    {
        g_coastStates.erase(nodeKey);
    }
    const double dt = gap>0 ? std::clamp(gap,1.0/500.0,0.25) : currentFrameIntervalSec();
    const auto telemetry=state.kalman.update(point.x,point.y,dt,0);
    state.time=observationTime;state.lookaheadSec=0;
    double ex = telemetry.estimate_x, ey = telemetry.estimate_y;
    if (NodeGraph::parameter(node, "humanizer_oneeuro") != 0.0)
    {
        // 1-Euro (or signal-dependent EMA when beta>0) after AimKalman2D:
        // fc = fcmin + beta * |v|.
        auto& f = g_oneEuroStates[nodeKey];
        const double fcMin = humanizer::nodeParam(node, "humanizer_oneeuro_fcmin", 1.0);
        const double beta = std::clamp(NodeGraph::parameter(node, "humanizer_oneeuro_beta"), 0.0, 5.0);
        const double dcut = humanizer::nodeParam(node, "humanizer_oneeuro_dcutoff", 1.0);
        const auto filtered = humanizer::oneEuroFilter(f, ex, ey, dt,
            std::clamp(fcMin, 0.05, 20.0), beta, std::clamp(dcut, 0.05, 20.0));
        ex = filtered.first; ey = filtered.second;
    }
    return {{ex,ey},{telemetry.velocity_x,telemetry.velocity_y},
        state.kalman.settings().velocity_damping,telemetry.prediction_ready,telemetry.warmup_remaining==0};
}

std::vector<std::pair<double,double>> MouseThread::graphFuturePositions(int nodeKey,int frames) const
{
    std::vector<std::pair<double,double>> positions;
    const auto found=graphPredictionStates.find(nodeKey);
    if(found==graphPredictionStates.end()||!found->second.kalman.initialized())return positions;
    frames=std::clamp(frames,0,300);positions.reserve(frames);
    const double dt=currentFrameIntervalSec();
    for(int i=1;i<=frames;++i) positions.push_back(found->second.kalman.predict(found->second.lookaheadSec+i*dt));
    return positions;
}

void MouseThread::retainGraphStates(const std::vector<int>& predictionKeys,const std::vector<int>& windKeys)
{
    for(auto it=graphPredictionStates.begin();it!=graphPredictionStates.end();)
        if(std::find(predictionKeys.begin(),predictionKeys.end(),it->first)==predictionKeys.end())it=graphPredictionStates.erase(it);else++it;
    for(auto it=graphWindStates.begin();it!=graphWindStates.end();)
        if(std::find(windKeys.begin(),windKeys.end(),it->first)==windKeys.end())it=graphWindStates.erase(it);else++it;
    for (auto it = g_oneEuroStates.begin(); it != g_oneEuroStates.end();)
        if (std::find(predictionKeys.begin(), predictionKeys.end(), it->first) == predictionKeys.end()) it = g_oneEuroStates.erase(it); else ++it;
    for (auto it = g_coastStates.begin(); it != g_coastStates.end();)
        if (std::find(predictionKeys.begin(), predictionKeys.end(), it->first) == predictionKeys.end()) it = g_coastStates.erase(it); else ++it;
    for (auto it = g_ouStates.begin(); it != g_ouStates.end();)
        if (std::find(windKeys.begin(), windKeys.end(), it->first) == windKeys.end()) it = g_ouStates.erase(it); else ++it;
    for (auto it = g_perlinClock.begin(); it != g_perlinClock.end();)
        if (std::find(windKeys.begin(), windKeys.end(), it->first) == windKeys.end()) it = g_perlinClock.erase(it); else ++it;
    for (auto it = g_overshootDebt.begin(); it != g_overshootDebt.end();)
        if (std::find(windKeys.begin(), windKeys.end(), it->first) == windKeys.end()) it = g_overshootDebt.erase(it); else ++it;
    for (auto it = g_overshootInitial.begin(); it != g_overshootInitial.end();)
        if (std::find(windKeys.begin(), windKeys.end(), it->first) == windKeys.end()) it = g_overshootInitial.erase(it); else ++it;
}

void MouseThread::sendMovementToDriver(int dx, int dy, std::uint64_t generation,
    double pixelScaleX, double pixelScaleY)
{
    if (dx == 0 && dy == 0)
    {
        return;
    }

    bool sent = false;
    detectionBuffer.withCurrentGeneration(generation, [&] {
        std::lock_guard<std::mutex> lock(inputDevicesMutex);

        if (workerStop || editorInputCaptured.load() || !mouseInput)
            return;

        sent = mouseInput->move(dx, dy);
    });

    if (sent)
    {
        recordMotionCompensationStep(dx, dy, pixelScaleX, pixelScaleY);
        appendWindDebugStep(dx, dy, pixelScaleX, pixelScaleY);
    }
}

void MouseThread::dispatchGraphRecoil(int counts,double pixelsPerCount)
{
    if(!std::isfinite(pixelsPerCount))return;
    sendMovementToDriver(0,counts,detectionGeneration,0,pixelsPerCount);
}

double MouseThread::graphSpeedCurve(double pointX,double pointY,NodeGraph::Point center,double maximumDistance,const NodeGraph::Node& node)
{
    const double distance=std::hypot(pointX-center.x,pointY-center.y);
    const double minimum=NodeGraph::parameter(node,"min_speed"),maximum=NodeGraph::parameter(node,"max_speed");
    const double snap=NodeGraph::parameter(node,"snap_radius"),nearRadius=NodeGraph::parameter(node,"near_radius");
    if(!std::isfinite(distance))return std::numeric_limits<double>::quiet_NaN();
    if(distance<snap)return minimum*NodeGraph::parameter(node,"snap_boost");
    if(nearRadius>0&&distance<nearRadius)
    {
        const double curve=1-std::pow(1-distance/nearRadius,NodeGraph::parameter(node,"exponent"));
        return minimum+(maximum-minimum)*curve;
    }
    return minimum+(maximum-minimum)*std::clamp(distance/std::max(1.0,maximumDistance),0.0,1.0);
}

NodeGraph::Movement MouseThread::graphMouseDelta(double pointX,double pointY,double speed,
    NodeGraph::Point center,const NodeGraph::Node& node)
{
    NodeGraph::Movement movement;
    // Deadzone pivot: ignore micro-offsets around the pivot so the aim does
    // not micro-jitter on already-centered targets. Disabled when 0.
    const double deadzonePx = std::clamp(NodeGraph::parameter(node,"humanizer_deadzone_px"),0.0,256.0);
    if (deadzonePx > 0.0 && std::isfinite(pointX) && std::isfinite(pointY) &&
        std::isfinite(center.x) && std::isfinite(center.y) &&
        std::hypot(pointX - center.x, pointY - center.y) < deadzonePx)
        return movement;
    const double fovX=NodeGraph::parameter(node,"fov_x"),fovY=NodeGraph::parameter(node,"fov_y");
    const double sens=NodeGraph::parameter(node,"sens"),yaw=NodeGraph::parameter(node,"yaw"),pitch=NodeGraph::parameter(node,"pitch");
    const double baseFov=NodeGraph::parameter(node,"base_fov");
    const double scale=NodeGraph::parameter(node,"fov_scaled")!=0&&baseFov>1?fovX/baseFov:1.0;
    if(!std::isfinite(pointX)||!std::isfinite(pointY)||!std::isfinite(speed)||
        !std::isfinite(center.x)||!std::isfinite(center.y)||sens==0||yaw==0||pitch==0||fovX<=0||fovY<=0)return movement;
    double dx=(pointX-center.x)*(fovX/std::max(1.0,screen_width))/(sens*yaw*scale)*speed;
    double dy=(pointY-center.y)*(fovY/std::max(1.0,screen_height))/(sens*pitch*scale)*speed;
    if(NodeGraph::parameter(node,"legacy_fps")!=0 && captureFps.load()>30)
    {const double correction=30.0/captureFps.load();dx*=correction;dy*=correction;}
    if(!std::isfinite(dx)||!std::isfinite(dy))return movement;
    const auto calibration=NodeGraph::mousePixelCalibration(node,{static_cast<double>(screen_width),static_cast<double>(screen_height)});
    movement.pixelScaleX=calibration.x;movement.pixelScaleY=calibration.y;
    movement.preciseCounts={dx,dy};movement.hasPreciseCounts=true;
    const int x=static_cast<int>(std::clamp(dx,-32767.0,32767.0)),y=static_cast<int>(std::clamp(dy,-32767.0,32767.0));
    if(x||y)movement.steps.push_back({x,y});
    return movement;
}

NodeGraph::Movement MouseThread::graphWindMotion(const NodeGraph::Movement& movement)
{
    if (movement.steps.size() > 32)
        return {};
    for (const auto& step : movement.steps)
        if (step.dx < -32767 || step.dx > 32767 || step.dy < -32767 || step.dy > 32767)
            return {};
    if (!wind_mouse_enabled || movement.steps.empty())
        return movement;

    // A movement value describes the desired relative displacement. Wind
    // synthesizes a bounded path for that displacement without dispatching it.
    long long dx = 0, dy = 0;
    for (const auto& step : movement.steps)
    {
        dx += step.dx;
        dy += step.dy;
    }
    return buildWindSteps(
        static_cast<int>(std::clamp(dx, -32767LL, 32767LL)),
        static_cast<int>(std::clamp(dy, -32767LL, 32767LL)));
}

NodeGraph::Movement MouseThread::graphWindMotion(int nodeKey, const NodeGraph::Movement& movement)
{
    auto found = graphWindStates.find(nodeKey);
    if (found == graphWindStates.end())
    {
        if (!hasGraphStateCapacity(graphWindStates, nodeKey))
            return {};
        // The first node inherits the reset baseline, preserving the existing
        // default motion profile. Further nodes receive independent phases/RNG.
        if (!graphWindStates.empty())
            resetWindScalars();
        found = graphWindStates.emplace(nodeKey, captureWindState()).first;
    }
    restoreWindState(found->second);
    const auto result = graphWindMotion(movement);
    // Invalid Wind parameters may reset all state; do not retain an iterator
    // across synthesis, since that reset clears the node map.
    graphWindStates.insert_or_assign(nodeKey, captureWindState());
    return result;
}

NodeGraph::Movement MouseThread::graphWindMotion(int nodeKey,const NodeGraph::Movement& movement,
    const NodeGraph::Node& node)
{
    wind_mouse_enabled=NodeGraph::parameter(node,"enabled")!=0;
    wind_G=NodeGraph::parameter(node,"wind_G");wind_W=NodeGraph::parameter(node,"wind_W");
    wind_M=NodeGraph::parameter(node,"wind_M");wind_D=NodeGraph::parameter(node,"wind_D");
    const humanizer::WindOpts opts = humanizer::windOptsFromNode(node);
    const WindOptsGuard guard(opts, nodeKey);
    auto resolved=graphWindMotion(nodeKey,movement);
    resolved.pixelScaleX=movement.pixelScaleX;resolved.pixelScaleY=movement.pixelScaleY;
    return resolved;
}

void MouseThread::dispatchGraphMovement(const NodeGraph::Movement& movement)
{
    // Graph evaluation already resolved coordinates, speed and optional wind.
    // Validate the complete batch before queueing any part of it.
    if (movement.steps.size() > 32)
        return;
    if(!std::isfinite(movement.pixelScaleX)||!std::isfinite(movement.pixelScaleY))return;
    for (const auto& step : movement.steps)
        if (step.dx < -32767 || step.dx > 32767 || step.dy < -32767 || step.dy > 32767)
            return;
    for (const auto& step : movement.steps)
        queueMove(step.dx, step.dy, movement.pixelScaleX, movement.pixelScaleY);
}

void MouseThread::clearQueuedMoves()
{
    std::lock_guard<std::mutex> dispatchLock(dispatchMutex);
    std::lock_guard<std::mutex> lock(queueMtx);
    ++queueGeneration;
    while (!moveQueue.empty())
        moveQueue.pop();
    resetWindScalars();
}

void MouseThread::pressMouse()
{
    if (workerStop || editorInputCaptured.load())
        return;
    if (!mouse_pressed)
    {
        detectionBuffer.withCurrentGeneration(detectionGeneration, [&] {
            std::lock_guard<std::mutex> lock(inputDevicesMutex);
            if (!workerStop && !editorInputCaptured.load() && mouseInput && mouseInput->leftDown())
                mouse_pressed = true;
        });
    }
}

void MouseThread::releaseMouse()
{
    if (mouse_pressed)
    {
        std::lock_guard<std::mutex> lock(inputDevicesMutex);
        if (!mouseInput)
        {
            mouse_pressed = false;
            return;
        }

        if (mouseInput->leftUp())
            mouse_pressed = false;
    }
}

void MouseThread::resetPrediction()
{
    graphPredictionStates.clear();
    graphWindStates.clear();
    clearQueuedMoves();

}

void MouseThread::storeFuturePositions(const std::vector<std::pair<double, double>>& positions)
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions = positions;
}

void MouseThread::clearFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions.clear();
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

void MouseThread::clearWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    windDebugTrail.clear();
    windDebugCursorX = center_x;
    windDebugCursorY = center_y;
}

std::vector<std::pair<double, double>> MouseThread::getWindDebugTrail()
{
    std::lock_guard<std::mutex> lock(windDebugTrailMutex);
    const auto now = std::chrono::steady_clock::now();
    pruneWindDebugTrailLocked(now);

    std::vector<std::pair<double, double>> out;
    out.reserve(windDebugTrail.size());
    for (const auto& p : windDebugTrail)
        out.emplace_back(p.x, p.y);
    return out;
}

void MouseThread::setMouseInput(IMouseInput* newMouseInput)
{
    clearQueuedMoves();
    std::lock_guard<std::mutex> lock(inputDevicesMutex);
    if (mouseInput != newMouseInput && mouse_pressed)
    {
        try
        {
            if (mouseInput)
                mouseInput->leftUp();
        }
        catch (...)
        {
            std::cerr << "[Mouse] Failed to release the old input device." << std::endl;
        }
        mouse_pressed = false;
    }
    mouseInput = newMouseInput;
}
