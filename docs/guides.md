# Usage and Troubleshooting Guides

| Guide | Contents |
| --- | --- |
| [First launch](guides/first-launch.md) | Build choice, models and initial setup |
| [Backends](guides/backends.md) | DML/CUDA requirements and checks |
| [Node graphs](guides/node-graphs.md) | Graph editing, controls and import/export |
| [Input methods](guides/input-methods.md) | Device setup and movement diagnostics |
| [Overlay](guides/overlay.md) | Preview, window placement and latency compensation |
| [UDP capture](guides/udp-capture.md) | Receiver setup and FFmpeg senders |
| [Capture diagnostics](guides/capture-diagnostics.md) | CPU/GPU capture counters |
| [Circle FOV](guides/circle-fov.md) | Circular target limits and display |
| [Data collection](guides/data-collection.md) | Saving frames and labels |
| [Recipes](guides/recipes.md) | Config examples and benchmarks |
| [Troubleshooting](guides/troubleshooting.md) | Diagnostic checklist |
| [Build workflow](guides/build-workflow.md) | Setup and local rebuilds |

References: [Configuration](config.md) · [Build from source](build.md).

## Overlay Productivity Notes

- Long lists (Inference `model_path`, depth models, COM ports, capture windows, node key codes, ONNX export source) have a search box at the top of the dropdown.
- The Overlay tab exposes Editor VSync and the in-game overlay FPS cap (`game_overlay_max_fps`, `0` = uncapped). The editor loop sleeps 10 ms normally and 16 ms while hidden or minimized.
- The Debug tab colors tracker confidences red (low) to green (high), shows the last per-frame inference time, and previews the circle-FOV ring.
